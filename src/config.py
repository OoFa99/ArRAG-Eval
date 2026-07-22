"""
Experiment configuration models using Pydantic.
Vector store: Qdrant. LLM backend: local Ollama (no API key required).
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import json
from pathlib import Path


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.
    
        Defaults:
            - domain: "general"
            - corpus_size: 5000
            - chunk_strategy: "fixed"
            - chunk_size: 512
            - chunk_overlap: 50
            - embedding_model: "intfloat/multilingual-e5-small"
            - vector_store_type: "qdrant"
            - qdrant_collection_name: "arrag-eval"
            - qdrant_url: "http://localhost:6333"
            - qdrant_api_key: None
            - retriever_type: "hybrid"
            - top_k: 10
            - rrf_k: 60
            - agentic: False
            - max_iterations: 2
            - language: "ar"
            - llm_model: "qwen3:4b"
            - ollama_host: "http://localhost:11434"
            - use_judge: True
            - num_test_samples: 20
            - experiment_name: "baseline-qdrant"
            - description: None
            - tags: []
    """
    
    # ============ Data & Corpus ============
    domain: Literal["general", "medical"] = Field(default="general")
    corpus_size: int = Field(default=5000, ge=100)
    
    # ============ Chunking ============
    chunk_strategy: Literal["fixed", "sentence", "semantic"] = Field(default="fixed")
    chunk_size: int = Field(default=512, ge=50, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    
    # ============ Embedding & Vector Store ============
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-small",
        description="HuggingFace embedding model"
    )
    
    vector_store_type: Literal["qdrant", "chroma"] = Field(
        default="qdrant",
        description="Vector database backend to use"
    )
    
    # === Qdrant Settings ===
    qdrant_collection_name: str = Field(default="arrag-eval")
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Local: http://localhost:6333 | Cloud: https://..."
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Required only for Qdrant Cloud"
    )
    
    # ============ Retrieval ============
    retriever_type: Literal["dense", "bm25", "hybrid"] = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60)
    
    # ============ Agentic & LLM ============
    agentic: bool = Field(default=False)
    max_iterations: int = Field(default=2, ge=1, le=5)
    language: Literal["ar", "en"] = Field(default="ar")
    
    # llm_model: str = Field(default="gemini-3.5-flash")
    
    # === Ollama Settings (local inference, no API key needed) ===
    llm_model: str = Field(
        default="qwen3:4b",
        description="Ollama model tag. qwen3:4b fits ~2.5GB VRAM at Q4_K_M "
                     "and has solid Arabic + English support — a good default "
                     "for 4GB-class GPUs (e.g. GTX 1650 Ti)."
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL. Default assumes a local `ollama serve`."
    )
    
    # ============ Evaluation ============
    use_judge: bool = Field(default=True)
    num_test_samples: int = Field(default=20, ge=5, le=500)
    
    # ============ Experiment Metadata ============
    experiment_name: str = Field(default="baseline-qdrant")
    description: Optional[str] = Field(default=None)
    tags: list = Field(default_factory=list)

    class Config:
        validate_assignment = True

    @validator('chunk_overlap')
    def overlap_less_than_chunk_size(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)

    def to_json(self) -> str:
        return self.model_dump_json(exclude_none=True)


# ========================= PRE-CONFIGURED CONFIGS =========================

BASELINE_CONFIG = ExperimentConfig(
    experiment_name="baseline-qdrant-local",
    vector_store_type="qdrant",
    qdrant_collection_name="arrag-eval",
    qdrant_url="http://localhost:6333",
    embedding_model="intfloat/multilingual-e5-small",
    retriever_type="hybrid",
    agentic=False,
    llm_model="qwen3:4b",
    ollama_host="http://localhost:11434",
    tags=["baseline", "qdrant", "local", "hybrid"]
)

# For Qdrant Cloud
QDRANT_CLOUD_CONFIG = ExperimentConfig(
    experiment_name="baseline-qdrant-cloud",
    vector_store_type="qdrant",
    qdrant_collection_name="arrag-eval-prod",
    qdrant_url="https://your-cluster.qdrant.io",   # Change this
    tags=["qdrant", "cloud"]
)

SEMANTIC_CONFIG = ExperimentConfig(
    experiment_name="semantic-qdrant-local",
    vector_store_type="qdrant",
    qdrant_collection_name="arrag-eval",
    qdrant_url="http://localhost:6333",
    embedding_model="intfloat/multilingual-e5-small",
    retriever_type="hybrid",
    chunk_strategy="semantic",
    agentic=False,
    llm_model="qwen3:4b",
    ollama_host="http://localhost:11434",
    tags=["semantic", "qdrant", "local", "hybrid"]
)

# ========================= ABLATION CONFIGS =========================
# Three retrieval configurations compared in experiments/run_ablation.py.
# Same corpus, same embedding model, same collection, same LLM — only
# retriever_type differs (and, for the agentic config, decomposition +
# a sufficiency-check loop layered on top of hybrid retrieval). Because
# only the query-time strategy changes, all three can safely query the
# same indexed Qdrant collection; there's no need to reindex per config.

DENSE_CONFIG = ExperimentConfig(
    experiment_name="ablation-dense",
    vector_store_type="qdrant",
    qdrant_collection_name="arrag-eval",
    qdrant_url="http://localhost:6333",
    embedding_model="intfloat/multilingual-e5-small",
    retriever_type="dense",
    agentic=False,
    llm_model="qwen3:4b",
    ollama_host="http://localhost:11434",
    tags=["ablation", "dense", "qdrant", "local"]
)

HYBRID_CONFIG = ExperimentConfig(
    experiment_name="ablation-hybrid",
    vector_store_type="qdrant",
    qdrant_collection_name="arrag-eval",
    qdrant_url="http://localhost:6333",
    embedding_model="intfloat/multilingual-e5-small",
    retriever_type="hybrid",
    agentic=False,
    llm_model="qwen3:4b",
    ollama_host="http://localhost:11434",
    tags=["ablation", "hybrid", "qdrant", "local"]
)

AGENTIC_CONFIG = ExperimentConfig(
    experiment_name="ablation-agentic",
    vector_store_type="qdrant",
    qdrant_collection_name="arrag-eval",
    qdrant_url="http://localhost:6333",
    embedding_model="intfloat/multilingual-e5-small",
    retriever_type="hybrid",
    agentic=True,
    llm_model="qwen3:4b",
    ollama_host="http://localhost:11434",
    tags=["ablation", "agentic",  "hybrid", "qdrant", "local"]
)