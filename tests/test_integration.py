"""
Minimal sanity check that the full pipeline (decompose → retrieve →
check_sufficiency → generate → evaluate) runs end-to-end without errors,
against a tiny 100-document index.

Requires locally running services:
  - Ollama:  `ollama serve`  (with `ollama pull qwen3:4b` done once)
  - Qdrant:  `docker run -p 6333:6333 qdrant/qdrant:latest`

If either isn't reachable, the test skips (pytest) or prints a warning
and returns (plain script) rather than failing — this is a pipeline
smoke test, not a check that your infra is running.

Run:
    python tests/test_integration.py
    pytest tests/test_integration.py -v
"""

import requests

from src.config import BASELINE_CONFIG
from src.data.corpus_download import load_wikipedia
from src.data.preprocessing import preprocess_corpus
from src.chunking.strategies import chunk_fixed_size
from src.agent.pipeline import ArRAGPipeline
from src.retrieval.qdrant_store import create_qdrant_store

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False


def _skip(msg: str):
    print(f"⚠ Skipping: {msg}")
    if _HAS_PYTEST:
        pytest.skip(msg)


def _ollama_reachable(host: str) -> bool:
    try:
        requests.get(host, timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def _qdrant_reachable(url: str) -> bool:
    try:
        requests.get(f"{url}/collections", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False


def setup_test_index(config):
    """Build a tiny, disposable Qdrant index for fast integration testing."""
    wiki = load_wikipedia(corpus_size=config.corpus_size)
    wiki_cleaned = preprocess_corpus(wiki["text"])

    chunks = []
    for doc in wiki_cleaned:
        chunks.extend(
            chunk_fixed_size(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        )

    # Separate collection name so this never collides with your real index.
    vector_store = create_qdrant_store(
        collection_name="arrag-eval-test",
        model_name=config.embedding_model,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )
    vector_store.add_documents(chunks, document_id="wikipedia_test_corpus")
    return vector_store


def test_end_to_end():
    """Test that the full pipeline runs without errors and returns a well-formed result."""
    config = BASELINE_CONFIG.copy(deep=True)
    config.corpus_size = 100  # tiny, for speed
    config.experiment_name = "integration-test"

    if not _ollama_reachable(config.ollama_host):
        _skip(f"Ollama not reachable at {config.ollama_host}. Run `ollama serve`.")
        return
    if not _qdrant_reachable(config.qdrant_url):
        _skip(
            f"Qdrant not reachable at {config.qdrant_url}. "
            f"Run `docker run -p 6333:6333 qdrant/qdrant:latest`."
        )
        return

    vector_store = setup_test_index(config)
    pipeline = ArRAGPipeline(config, vector_store)

    question = "ما هو الذكاء الاصطناعي؟"
    result = pipeline.run(question)

    # ---- Structural checks ----
    assert "answer" in result
    assert "retrieved_chunks" in result
    assert "evaluation_scores" in result
    assert "sub_queries" in result

    # ---- Content checks ----
    assert len(result["answer"]) > 0, "Answer should not be empty"
    assert isinstance(result["retrieved_chunks"], list)
    assert len(result["retrieved_chunks"]) > 0, "Should retrieve at least one chunk"

    print("✓ Integration test passed")
    print(f"  Question:         {question}")
    print(f"  Sub-queries:      {result['sub_queries']}")
    print(f"  Chunks retrieved: {len(result['retrieved_chunks'])}")
    print(f"  Answer preview:   {result['answer'][:150]}...")


if __name__ == "__main__":
    test_end_to_end()