"""
Query decomposition module for breaking complex questions into atomic sub-queries.

This module handles:
- Detecting whether decomposition is needed (simple vs. complex questions)
- Using a local Ollama LLM to decompose complex questions into sub-queries
- JSON-mode generation (no more markdown-fence / regex parsing)
- Fallback behavior for malformed outputs
"""

from src.agent.ollama_client import generate_json, DEFAULT_MODEL, OllamaGenerationError
from typing import List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Enum for query complexity levels."""
    SIMPLE = "simple"        # Single-hop factoid question
    COMPLEX = "complex"      # Multi-hop or conditional question


def detect_query_complexity(query: str) -> QueryComplexity:
    """
    Heuristically detect if a query is simple or complex.
    
    Complex indicators:
    - Multiple questions (how/what/why/when combined)
    - Conditional words (if, when, before, after, during)
    - Comparison words (vs, compared to, difference between)
    - Multi-entity references (several named entities)
    
    Args:
        query (str): Input question
    
    Returns:
        QueryComplexity: SIMPLE or COMPLEX
    
    Notes:
        - This is a heuristic, not perfect
        - Simple queries skip decomposition (faster, cheaper)
        - Complex queries trigger LLM decomposition
    
    Example:
        >>> detect_query_complexity("What is machine learning?")
        <QueryComplexity.SIMPLE: 'simple'>
        
        >>> detect_query_complexity("How did machine learning evolve and what are its applications?")
        <QueryComplexity.COMPLEX: 'complex'>
    """
    # Count question words (in English and Arabic)
    question_words = [
        'how', 'what', 'when', 'where', 'why', 'which',
        'كيف', 'ماذا', 'متى', 'أين', 'لماذا', 'أي'
    ]
    question_count = sum(1 for word in question_words if word in query.lower())
    
    # Conditional indicators
    conditional_words = [
        'if', 'when', 'before', 'after', 'during', 'while',
        'إذا', 'عندما', 'قبل', 'بعد', 'خلال', 'بينما'
    ]
    has_conditional = any(word in query.lower() for word in conditional_words)
    
    # Comparison indicators
    comparison_words = [
        'vs', 'versus', 'compared', 'difference', 'contrast',
        'مقابل', 'مقارنة', 'الفرق', 'بخلاف'
    ]
    has_comparison = any(word in query.lower() for word in comparison_words)
    
    # Conjunction indicators (multiple clauses)
    has_conjunction = ' and ' in query.lower() or 'و' in query
    
    # Decide complexity
    complexity_score = question_count + has_conditional + has_comparison + has_conjunction
    
    if complexity_score >= 2:
        return QueryComplexity.COMPLEX
    else:
        return QueryComplexity.SIMPLE


class QueryDecomposer:
    """
    Decomposes complex questions into atomic sub-queries using a local Ollama LLM.
    
    Example:
        >>> decomposer = QueryDecomposer(api_key="sk-...", language="ar")
        >>> question = "ما هي السياسات التي أثرت على البطالة في مصر بعد 2011؟"
        >>> sub_queries = decomposer.decompose(question)
        >>> print(sub_queries)
        ['ما هي السياسات الاقتصادية المصرية بعد 2011؟',
         'كيف تأثرت معدلات البطالة على الثورة المصرية؟',
         'ما هي الإصلاحات التي طبقتها مصر؟']
    """
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL,
                 host: Optional[str] = None,
                 language: str = "ar",
                 max_retries: int = 2):
        """
        Initialize the QueryDecomposer.
        
        Args:
            model (str): Ollama model tag to use. Defaults to "qwen3:4b"
                        (fits comfortably in 4GB VRAM).
            host (Optional[str]): Ollama server URL. Defaults to the
                        ollama_client module default (http://localhost:11434).
            language (str): Language of queries. "ar" for Arabic, "en" for English
            max_retries (int): Number of retries on generation/parse failure
        """
        self.model = model
        self.host = host
        self.language = language
        self.max_retries = max_retries
        
        # Language-specific prompts
        self.prompts = self._get_language_prompts()
    
    def _get_language_prompts(self) -> dict:
        """Get prompts for the configured language."""
        if self.language == "ar":
            return {
                "system": """أنت مساعد متخصص في تحليل الأسئلة العربية المعقدة.
مهمتك تقسيم الأسئلة المعقدة متعددة الأجزاء إلى أسئلة بسيطة ذرية، 
بحيث يمكن الإجابة على كل سؤال من نص واحد.

قواعد:
- أنتج 2-4 أسئلة فقط (أقل تعقيداً = عدد أقل من الأسئلة)
- كل سؤال يجب أن يكون قابلاً للإجابة من فقرة واحدة
- حافظ على السياق والمعنى الأصلي
- الرد فقط بـ JSON صحيح، بدون نص إضافي""",
                
                "user": """قسّم هذا السؤال إلى أسئلة فرعية بسيطة:
السؤال: {question}

الرد بصيغة JSON فقط:
{{"sub_queries": ["السؤال الفرعي 1", "السؤال الفرعي 2", ...]}}"""
            }
        else:  # English
            return {
                "system": """You are a question decomposition assistant for English text.
Given a complex multi-hop question, decompose it into 2-4 simple atomic sub-questions,
each answerable from a single passage.

Rules:
- Produce 2-4 sub-questions (simpler = fewer questions)
- Each sub-question must be answerable from one paragraph
- Preserve the original context and meaning
- Respond ONLY with valid JSON, no additional text""",
                
                "user": """Decompose this question into simple sub-questions:
Question: {question}

Respond with ONLY valid JSON:
{{"sub_queries": ["sub-question 1", "sub-question 2", ...]}}"""
            }
    
    def _validate_sub_queries(self, parsed: Optional[dict]) -> Optional[List[str]]:
        """
        Validate and clean a parsed JSON response into a sub-queries list.

        Args:
            parsed (Optional[dict]): Already-parsed JSON dict from the LLM
                        (or None if generation/parsing failed upstream).

        Returns:
            Optional[List[str]]: Cleaned sub-queries, or None if invalid/empty.
        """
        if not parsed:
            return None

        sub_queries = parsed.get("sub_queries", [])

        # Validate: must be a list of strings
        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            # Filter empty strings
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            if sub_queries:
                return sub_queries

        return None
    
    def decompose(self, 
                  query: str, 
                  auto_detect: bool = True) -> List[str]:
        """
        Decompose a question into sub-queries.
        
        If auto_detect=True, first checks if decomposition is needed.
        Simple queries are returned as-is (list with one element).
        Complex queries are decomposed via LLM.
        
        Args:
            query (str): The question to decompose
            auto_detect (bool): Whether to auto-detect query complexity. 
                              If False, always use LLM decomposition.
        
        Returns:
            List[str]: List of sub-queries (usually 1-4 queries)
        
        Raises:
            ValueError: If decomposition fails after max_retries
        
        Example:
            >>> decomposer = QueryDecomposer()
            >>> decomposer.decompose("What is machine learning and its applications?")
            ['What is machine learning?', 'What are applications of machine learning?']
        """
        # Auto-detect complexity
        if auto_detect:
            complexity = detect_query_complexity(query)
            if complexity == QueryComplexity.SIMPLE:
                logger.info(f"Query detected as SIMPLE, returning as-is: {query}")
                return [query]
        
        # Attempt decomposition with retries
        kwargs = {"model": self.model}
        if self.host:
            kwargs["host"] = self.host

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Decomposing query (attempt {attempt + 1}/{self.max_retries}): {query}")

                parsed = generate_json(
                    prompt=self.prompts["user"].format(question=query),
                    system_instruction=self.prompts["system"],
                    max_tokens=500,
                    temperature=0.1,
                    **kwargs,
                )

                sub_queries = self._validate_sub_queries(parsed)
                if sub_queries:
                    logger.info(f"Successfully decomposed into {len(sub_queries)} sub-queries")
                    return sub_queries

                # Parse/validation failed, will retry
                logger.warning(f"Failed to get valid sub_queries from response: {parsed}")

            except OllamaGenerationError as e:
                logger.error(f"Ollama error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        # All retries failed — fallback to returning original query
        logger.warning(f"Decomposition failed after {self.max_retries} attempts, returning original query")
        return [query]


# Convenience function for one-off decomposition
def decompose_query(query: str, 
                    language: str = "ar",
                    model: str = DEFAULT_MODEL,
                    auto_detect: bool = True) -> List[str]:
    """
    Decompose a single query without creating a QueryDecomposer instance.
    
    Args:
        query (str): Question to decompose
        language (str): "ar" or "en"
        model (str): Ollama model tag to use. Defaults to "qwen3:4b"
        auto_detect (bool): Whether to auto-detect complexity
    
    Returns:
        List[str]: Sub-queries
    
    Example:
        >>> sub_queries = decompose_query("ما هي أنواع الذكاء الاصطناعي؟")
    """
    decomposer = QueryDecomposer(language=language)
    return decomposer.decompose(query, auto_detect=auto_detect)