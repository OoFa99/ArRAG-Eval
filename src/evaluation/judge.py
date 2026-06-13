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
        