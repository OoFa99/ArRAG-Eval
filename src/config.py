"""
Experiment configuration models using Pydantic.
Now supports Qdrant as the primary vector store.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal
import json
from pathlib import Path


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.
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
    
    vector_store_type: Literal["qdrant", "pinecone", "chroma"] = Field(
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
    
    # === Legacy Pinecone (for easy switching) ===
    pinecone_index_name: Optional[str] = Field(default=None)
    pinecone_api_key: Optional[str] = Field(default=None)
    
    # ============ Retrieval ============
    retriever_type: Literal["dense", "bm25", "hybrid"] = Field(default="hybrid")
    top_k: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60)
    
    # ============ Agentic & LLM ============
    agentic: bool = Field(default=False)
    max_iterations: int = Field(default=2, ge=1, le=5)
    language: Literal["ar", "en"] = Field(default="ar")
    llm_model: str = Field(default="gemini-3.5-flash")
    
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

# Keep Pinecone if you want to switch back
PINECONE_CONFIG = ExperimentConfig(
    experiment_name="baseline-pinecone",
    vector_store_type="pinecone",
    pinecone_index_name="arrag-eval",
    tags=["pinecone"]
)