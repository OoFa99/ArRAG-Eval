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