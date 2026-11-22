"""Live smoke test using a small, repository-owned Arabic corpus."""

from contextlib import closing
import os
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from src.config import BASELINE_CONFIG


CORPUS = Path(__file__).with_name("fixtures") / "smoke_corpus.md"


def require_service(name, url):
    try:
        response = requests.get(url, timeout=5, allow_redirects=False)
    except requests.RequestException as error:
        pytest.fail(f"{name} is unreachable at {url}: {error}", pytrace=False)
    assert response.status_code == 200, f"{name} at {url} returned HTTP {response.status_code}"


@pytest.mark.integration
def test_end_to_end(pytestconfig):
    if not pytestconfig.getoption("--run-integration"):
        pytest.skip("Pass --run-integration to run the live pipeline smoke test.")

    config = BASELINE_CONFIG.model_copy()
    config.llm_model = os.getenv("ARRAG_SMOKE_OLLAMA_MODEL") or config.llm_model
    require_service("Ollama", f"{config.ollama_host}/api/tags")
    require_service("Qdrant", f"{config.qdrant_url}/collections")

    from src.agent.pipeline import ArRAGPipeline
    from src.chunking.strategies import chunk_fixed_size
    from src.data.preprocessing import preprocess_text
    from src.retrieval.qdrant_store import create_qdrant_store

    chunks = chunk_fixed_size(
        preprocess_text(CORPUS.read_text(encoding="utf-8")),
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    collection_name = f"arrag-eval-smoke-{uuid4().hex}"
    vector_store = create_qdrant_store(
        collection_name=collection_name,
        model_name=config.embedding_model,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )
    with closing(vector_store.client) as client:
        try:
            vector_store.add_documents(chunks, document_id="smoke_corpus")
            result = ArRAGPipeline(config, vector_store).run("ما هي عاصمة مصر؟")

            assert any("القاهرة" in chunk["text"] for chunk in result["retrieved_chunks"])
            assert result["answer"].strip(), "Ollama returned an empty answer"
            assert result["answer"] != "Error: Could not generate answer"
        finally:
            client.delete_collection(collection_name)
