"""
LLM-as-Judge: Use an LLM to make evaluation decisions.

This module handles:
- Decomposing answers into claims
- Checking if claims are supported by context
- Assessing chunk relevance
- Generating questions from answers
- Caching results to avoid re-computation
- Arabic-specific prompts and handling

Why LLM-as-Judge?
  Some things are hard to compute algorithmically:
  - Is this claim really supported by the context?
  - Does this chunk contribute to the answer?
  
  LLMs are good at these judgments. We use them strategically,
  cache the results, and measure their confidence.
"""

import json
import logging
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from openai import OpenAI, APIError

logger = logging.getLogger(__name__)

class JudgeCache:
    """
    SQLite-based cache for judge decisions.
    
    Avoids re-paying for the same LLM calls across experiments.
    
    Schema:
    - query_hash: hash(query) → fast lookup
    - decision_type: "is_supported", "is_relevant", "decompose_claims", etc.
    - result: JSON-serialized result
    """
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory for cache database
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / "judge_cache.db"
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite database and table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS judge_cache (
                    query_hash TEXT PRIMARY KEY,
                    decision_type TEXT,
                    input_text TEXT,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
    def _hash_query(self, decision_type: str, input_text: str) -> str:
        """Create a hash for the query."""
        combined = f"{decision_type}:{input_text}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, decision_type: str, input_text: str) -> Optional[Any]:
        """
        Retrieve cached result.
        
        Args:
            decision_type: Type of decision (e.g., "is_supported")
            input_text: The input (claim, chunk, etc.)
        
        Returns:
            Cached result or None
        """
        query_hash = self._hash_query(decision_type, input_text)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT result FROM judge_cache
                WHERE query_hash = ? AND decision_type = ?
            """, (query_hash, decision_type)
            )
            
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        
        return None
    
    def set(self, decision_type: str, input_text: str, result: Any) -> None:
        """
        Store result in cache.
        
        Args:
            decision_type: Type of decision (e.g., "is_supported")
            input_text: The input (claim, chunk, etc.)
            result: The result to cache
        """
        query_hash = self._hash_query(decision_type, input_text)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO judge_cache 
                (query_hash, decision_type, input_text, result)
                VALUES (?, ?, ?, ?)
            """, (query_hash, decision_type, input_text, json.dumps(result))
            )
            conn.commit()
            
    def clear(self) -> None:
        """Clear the cache."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM judge_cache")
            conn.commit()
            
class LLMJudge:
    """
    Use an LLM to make evaluation decisions for RAG quality.
    
    Decision types:
    - is_claim_supported: Is a claim supported by context?
    - is_chunk_relevant: Does a chunk contribute to the answer?
    - decompose_claims: Break answer into atomic claims
    - generate_questions: Generate questions from an answer
    
    Caching:
    - Each decision is cached by (decision_type, input)
    - Subsequent calls retrieve from cache
    - Significant cost savings across experiments
    
    Arabic Support:
    - Arabic-specific prompts for better reasoning
    - Handles right-to-left text
    - Aware of dialectal variation
    """
    
    def __init__(self,
                 api_key: str,
                 model: str = "gpt-40-mini",
                 language: str = "ar",
                 use_cache: bool = True,
                 cache_dir: str = ".cache"):
        """
        Initialize LLM judge.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-4o-mini or gpt-4o recommended)
            language: "ar" for Arabic, "en" for English
            use_cache: Whether to cache judge decisions
            cache_dir: Directory for cache database
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language
        self.use_cache = use_cache
        
        if use_cache:
            self.cache = JudgeCache(cache_dir=cache_dir)
        else:
            self.cache = None
        
        self.prompts = self._get_prompts()
        
    def _get_prompts(self) -> Dict[str, str]:
        """Get language-specific prompts."""
        if self.language == "ar":
            return {
                "claim_support": """أنت مقيم متخصص في التحقق من صحة المعلومات.

السؤال: {question}
الادعاء: "{claim}"
السياق المسترجع:
{context}

سؤال: هل الادعاء مدعوم بالكامل من السياق؟

ملاحظات:
- إذا كان الادعاء يتطابق مع محتوى السياق بشكل صحيح → نعم
- إذا كان الادعاء يتعارض مع السياق → لا
- إذا كان الادعاء غير مذكور في السياق → لا
- الادعاءات الجزئية المدعومة → لا (يجب الدعم الكامل)

الرد بصيغة JSON فقط:
{{"supported": true/false, "reasoning": "السبب المختصر"}}""",
                
                "chunk_relevance": """أنت محلل لملاءمة المعلومات.

السؤال الأصلي: {question}
الإجابة المولدة: {answer}
الفقرة المسترجعة:
{chunk}

سؤال: هل هذه الفقرة ساهمت في الإجابة على السؤال؟

ملاحظات:
- الفقرات ذات الصلة تحتوي على معلومات استخدمت في الإجابة
- الفقرات غير ذات الصلة لا تساعد في الإجابة على السؤال

الرد بصيغة JSON فقط:
{{"relevant": true/false, "reasoning": "لماذا أو لماذا لا"}}""",
                
                "decompose_claims": """أنت متخصص في تحليل الجمل المعقدة.

الإجابة: "{answer}"

قسّم الإجابة إلى ادعاءات ذرية بسيطة (مستقلة عن بعضها).

قواعد:
- كل ادعاء يجب أن يكون جملة واحدة بسيطة
- تجنب الادعاءات المركبة (استخدم و/أو منفصلة)
- شمل الأرقام والحقائق الملموسة
- لا تشمل العبارات الزائدة

الرد بصيغة JSON فقط:
{{"claims": ["الادعاء 1", "الادعاء 2", ...]}}""",
                
                "generate_questions": """أنت خبير في صياغة الأسئلة.

الإجابة: "{answer}"

اكتب {num_questions} أسئلة مختلفة التي يمكن أن تحصل على هذه الإجابة.

قواعد:
- الأسئلة يجب أن تكون أسئلة طبيعية بالعربية
- يجب أن تحصل الإجابة على كل سؤال
- اجعلها متنوعة (طرق مختلفة لسؤال نفس الشيء)

الرد بصيغة JSON فقط:
{{"questions": ["السؤال 1", "السؤال 2", ...]}}"""
            }
        else:  # English
            return {
                "claim_support": """You are a fact-checking expert.

Question: {question}
Claim: "{claim}"
Retrieved context:
{context}

Question: Is the claim fully supported by the context?

Notes:
- If claim matches context correctly → yes
- If claim contradicts context → no
- If claim is not mentioned → no
- Partially supported claims → no (must be fully supported)

Respond ONLY with valid JSON:
{{"supported": true/false, "reasoning": "brief reason"}}""",
                
                "chunk_relevance": """You are an information relevance analyzer.

Original question: {question}
Generated answer: {answer}
Retrieved chunk:
{chunk}

Question: Did this chunk contribute to answering the question?

Notes:
- Relevant chunks contain information used in the answer
- Irrelevant chunks don't help answer the question

Respond ONLY with valid JSON:
{{"relevant": true/false, "reasoning": "why or why not"}}""",
                
                "decompose_claims": """You are a sentence decomposition expert.

Answer: "{answer}"

Decompose the answer into simple atomic claims (independent of each other).

Rules:
- Each claim should be one simple sentence
- Avoid compound claims (use separate and/or)
- Include numbers and concrete facts
- Exclude filler phrases

Respond ONLY with valid JSON:
{{"claims": ["claim 1", "claim 2", ...]}}""",
                
                "generate_questions": """You are a question generation expert.

Answer: "{answer}"

Write {num_questions} different questions that could result in this answer.

Rules:
- Questions should be natural English questions
- The answer should address each question
- Make them varied (different ways to ask the same thing)

Respond ONLY with valid JSON:
{{"questions": ["question 1", "question 2", ...]}}"""
            }
            
    def _call_llm(self, prompt: str, retries: int = 2) -> Optional[str]:
        """
        Call the LLM with retry logic.
            
        Args:
            prompt: The prompt to send
            retries: Number of retry attempts
            
        Returns:
            LLM response or None if failed
        """
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except APIError as e:
                logger.error(f"LLM API error on attempt {attempt + 1}: {e}")
                if attempt == retries - 1:
                    return None
        return None

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """
        Parse LLM response as JSON.
        
        Args:
            response: The raw LLM response string
        
        Returns:
            Parsed JSON as dict or None if parsing fails
        """
        if not response:
            return None
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {response[:200]}")
                    return None
        
        return None

    def is_claim_supported(self, claim: str, context: str) -> bool:
        """
        Determine if a claim is supported by context.
            
        Args:
            claim: The claim to check
            context: The retrieved context
            
        Returns:
            True if supported, False otherwise
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get("is_claim_supported", f"{claim}|{context[:500]}")
            if cached is not None:
                return cached.get("supported", False)
            
        # Generate prompt
        prompt = self.prompts["claim_support"].format(
            question="",
            claim=claim,
            context=context
        )
        
        # Call LLM
        response = self._call_llm(prompt)
        parsed = self._parse_json_response(response)
        
        if parsed is None:
            logger.warning(f"Failed to parse claim support response")
            return False
        
        supported = parsed.get("supported", False)
        
        # Cache result
        if self.cache:
            self.cache.set(
                "is_claim_supported",
                f"{claim}|{context[:500]}",
                {"supported": supported}
            )
        
        return supported
    
    def is_chunk_relevant(self, chunk: str, question: str, answer: str) -> bool:
        """
        Determine if a chunk is relevant to the answer.
        
        Args:
            question: The question
            chunk: The retrieved chunk
            answer: The generated answer
        
        Returns:
            True if relevant, False otherwise
        """