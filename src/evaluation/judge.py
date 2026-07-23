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

from src.agent.ollama_client import generate_json, DEFAULT_MODEL, OllamaGenerationError

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
                 model: str = DEFAULT_MODEL,
                 host: Optional[str] = None,
                 language: str = "ar",
                 use_cache: bool = True,
                 cache_dir: str = ".cache"):
        """
        Initialize LLM judge.
        
        Args:
            model: Ollama model tag to use. Defaults to "qwen3:4b".
            host: Ollama server URL. Defaults to the ollama_client module
                  default (http://localhost:11434).
            language: "ar" for Arabic, "en" for English
            use_cache: Whether to cache judge decisions
            cache_dir: Directory for cache database
        """
        self.model = model
        self.host = host
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

الرد بصيغة JSON فقط. حقل "reasoning" يجب ألا يتجاوز 10 كلمات:
{{"supported": true/false, "reasoning": "سبب مختصر جدا (10 كلمات كحد أقصى)"}}""",
                
                "chunk_relevance": """أنت محلل لملاءمة المعلومات.

السؤال الأصلي: {question}
الإجابة المولدة: {answer}
الفقرة المسترجعة:
{chunk}

سؤال: هل هذه الفقرة ساهمت في الإجابة على السؤال؟

ملاحظات:
- الفقرات ذات الصلة تحتوي على معلومات استخدمت في الإجابة
- الفقرات غير ذات الصلة لا تساعد في الإجابة على السؤال

الرد بصيغة JSON فقط. حقل "reasoning" يجب ألا يتجاوز 10 كلمات:
{{"relevant": true/false, "reasoning": "سبب مختصر جدا (10 كلمات كحد أقصى)"}}""",
                
                "decompose_claims": """أنت متخصص في تحليل الجمل المعقدة.

الإجابة: "{answer}"

قسّم الإجابة إلى ادعاءات ذرية بسيطة (مستقلة عن بعضها)، بحد أقصى 6 ادعاءات.

قواعد:
- كل ادعاء يجب أن يكون جملة واحدة بسيطة
- تجنب الادعاءات المركبة (استخدم و/أو منفصلة)
- شمل الأرقام والحقائق الملموسة
- لا تشمل العبارات الزائدة
- 6 ادعاءات كحد أقصى — إذا كانت الإجابة تحتوي على أكثر من ذلك، اختر الأهم

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

Respond ONLY with valid JSON. The "reasoning" field must be 10 words or fewer:
{{"supported": true/false, "reasoning": "very brief reason (max 10 words)"}}""",
                
                "chunk_relevance": """You are an information relevance analyzer.

Original question: {question}
Generated answer: {answer}
Retrieved chunk:
{chunk}

Question: Did this chunk contribute to answering the question?

Notes:
- Relevant chunks contain information used in the answer
- Irrelevant chunks don't help answer the question

Respond ONLY with valid JSON. The "reasoning" field must be 10 words or fewer:
{{"relevant": true/false, "reasoning": "very brief reason (max 10 words)"}}""",
                
                "decompose_claims": """You are a sentence decomposition expert.

Answer: "{answer}"

Decompose the answer into simple atomic claims (independent of each other), at most 6 claims.

Rules:
- Each claim should be one simple sentence
- Avoid compound claims (use separate and/or)
- Include numbers and concrete facts
- Exclude filler phrases
- Maximum 6 claims — if the answer has more, keep only the most important ones

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

            
    def _call_llm(self, prompt: str, max_tokens: int = 400) -> Optional[Dict[str, Any]]:
        """
        Call the local LLM and parse its response as JSON.

        Uses Ollama's native JSON mode (format="json"), so the model is
        constrained to emit valid JSON — retries on transient failures are
        handled inside OllamaClient itself, so this is a thin wrapper.

        Args:
            prompt: The prompt to send
            max_tokens: Token budget for this specific decision type.
                Callers pass a value sized to what the response actually
                needs to contain (see class docstring).

        Returns:
            Parsed JSON dict, or None if generation/parsing failed
        """        
        kwargs = {"model": self.model}
        if self.host:
            kwargs["host"] = self.host

        try:
            return generate_json(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.1,
                **kwargs,
            )
        except OllamaGenerationError as e:
            logger.error(f"Ollama call failed: {e}")
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
        parsed = self._call_llm(prompt)
        
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
        # Check cache
        if self.cache:
            cache_key = f"{question}|{chunk[:500]}|{answer[:500]}"
            cached = self.cache.get("is_chunk_relevant", cache_key)
            if cached is not None:
                return cached.get("relevant", False)
        
        # Format prompt
        prompt = self.prompts["chunk_relevance"].format(
            question=question,
            answer=answer,
            chunk=chunk
        )
        
        # Call LLM
        parsed = self._call_llm(prompt)
        
        if parsed is None:
            logger.warning(f"Failed to parse chunk relevance response")
            return False
        
        relevant = parsed.get("relevant", False)
        
        # Cache result
        if self.cache:
            self.cache.set(
                "is_chunk_relevant",
                f"{question}|{chunk[:500]}|{answer[:500]}",
                {"relevant": relevant}
            )
        
        return relevant
    
    def decompose_claims(self, answer: str) -> List[str]:
        """
        Break an answer into atomic claims.
        
        Args:
            answer: The answer to decompose
        
        Returns:
            List of claims
        """
        # Check cache
        if self.cache:
            cached = self.cache.get("decompose_claims", answer[:500])
            if cached is not None:
                return cached.get("claims", [])
            
        # Format prompt
        prompt = self.prompts["decompose_claims"].format(answer=answer)
        
        # Call LLM
        parsed = self._call_llm(prompt, max_tokens=800)
        
        if parsed is None:
            logger.warning(f"Failed to decompose answer into claims")
            return [answer]  # Fallback: return as single claim
        
        claims = parsed.get("claims", [answer])
        
        # Cache result
        if self.cache:
            self.cache.set(
                "decompose_claims",
                answer[:500],
                {"claims": claims}
            )
        
        return claims
    
    def generate_questions(self, answer: str, num_questions: int = 3) -> List[str]:
        """
        Generate questions that could result in this answer.
        
        Args:
            answer: The answer
            num_questions: How many questions to generate
        
        Returns:
            List of generated questions
        """
        # Check cache
        if self.cache:
            cache_key = f"{answer[:500]}|num={num_questions}"
            cached = self.cache.get("generate_questions", cache_key)
            if cached is not None:
                return cached.get("questions", [])
        
        # Format prompt
        prompt = self.prompts["generate_questions"].format(
            answer=answer,
            num_questions=num_questions
        )
        
        # Call LLM
        parsed = self._call_llm(prompt)
        
        if parsed is None:
            logger.warning(f"Failed to generate questions")
            return []
        
        questions = parsed.get("questions", [])
        
        # Cache result
        if self.cache:
            self.cache.set(
                "generate_questions",
                f"{answer[:500]}|num={num_questions}",
                {"questions": questions}
            )
        
        return questions
    