"""
Same as run_baseline.py, but wraps the run in an MLFlow run so config +
results are tracked and comparable across future runs (the ablation
studies in Phase 3 build directly on this).

Prereqs (unchanged from run_baseline.py):
  - `ollama serve` running, with `ollama pull qwen3:4b` done once
  - Qdrant running: docker run -p 6333:6333 qdrant/qdrant:latest
  - pip install mlflow (if not already installed)

Run:
    python experiments/run_baseline_mlflow.py
    mlflow ui   # in a separate terminal, to view the logged run
"""

import os
import json
import mlflow
from dotenv import load_dotenv

from src.config import BASELINE_CONFIG
from src.data.corpus_download import load_arcd, load_wikipedia
from src.data.preprocessing import preprocess_corpus
from src.chunking.strategies import chunk_fixed_size
from src.agent.pipeline import ArRAGPipeline
from src.retrieval.qdrant_store import create_qdrant_store

# Only needed for optional QDRANT_API_KEY (cloud Qdrant), same as run_baseline.py
load_dotenv()

MLFLOW_EXPERIMENT_NAME = "ArRAG-Eval"
OUTPUT_PATH = "baseline_outputs.json"


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Work on a copy so this run never mutates the shared BASELINE_CONFIG
    # singleton — matters once run_ablation.py starts reusing it (Phase 3).
    config = BASELINE_CONFIG.copy(deep=True)
    config.corpus_size = 500
    config.num_test_samples = 20

    with mlflow.start_run(run_name="week1-baseline-test"):
        # Log config immediately, so even a crashed run leaves a record.
        # Cast every value to str: mlflow params must be string-serializable,
        # and config.tags is a list.
        mlflow.log_params({k: str(v) for k, v in config.to_dict().items()})

        # ---- 1. Load data ----
        arcd = load_arcd()
        wiki = load_wikipedia(corpus_size=config.corpus_size)

        # ---- 2. Preprocess ----
        wiki_cleaned = preprocess_corpus(wiki["text"])

        # ---- 3. Chunk ----
        chunks = []
        for doc in wiki_cleaned:
            chunks.extend(
                chunk_fixed_size(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
            )
        mlflow.log_metric("num_chunks", len(chunks))

        # ---- 4. Index into Qdrant ----
        vector_store = create_qdrant_store(
            collection_name=config.qdrant_collection_name,
            model_name=config.embedding_model,
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or os.getenv("QDRANT_API_KEY"),
        )
        vector_store.add_documents(chunks, document_id="wikipedia_corpus")

        # ---- 5. Pipeline (local Ollama — no api_key) ----
        print(f"Using Ollama model: {config.llm_model} @ {config.ollama_host}")
        pipeline = ArRAGPipeline(config=config, vector_store=vector_store)

        # ---- 6. Run ----
        test_data = arcd["validation"][: config.num_test_samples]
        results = pipeline.run_batch(test_data["question"])

        # ---- 7. Log a sample result + full outputs as MLFlow artifacts ----
        mlflow.log_dict(results[0], "sample_result.json")

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(OUTPUT_PATH)

        # ---- 8. A couple of headline run-level metrics worth tracking now ----
        mlflow.log_metric("num_questions", len(results))
        if results:
            mlflow.log_metric(
                "pct_decomposed",
                sum(r["is_decomposed"] for r in results) / len(results),
            )

        print("Qdrant Stats:", vector_store.get_stats())

    print(f"✓ MLFlow run logged. {OUTPUT_PATH} written. Check results with: mlflow ui")


if __name__ == "__main__":
    main()