"""
Runs the same test questions through three retrieval configurations —
dense, hybrid (RRF), and agentic — and logs each as its own MLFlow run
so they're directly comparable in the MLFlow UI or via
scripts/create_ablation_table.py.

PREREQUISITE: this depends on retrieve_node in src/agent/pipeline.py
dispatching on config.retriever_type / config.top_k instead of hardcoding
hybrid_query(top_k=10). If you haven't applied that patch, every config
below retrieves identically and this ablation is meaningless — check
that retrieve_node has the if/elif on config.retriever_type before running
this.

All three configs point at the same Qdrant collection and embedding
model, so the corpus is indexed once — retrieval strategy is a query-time
choice (dense/bm25/hybrid), not an indexing-time one. The agentic config
also retrieves via hybrid search per sub-query; what it adds on top is
query decomposition and a sufficiency-check loop.

Prereqs:
  - `ollama serve` running, with `ollama pull qwen3:4b` done once
  - Qdrant running: docker run -p 6333:6333 qdrant/qdrant:latest
  - pip install mlflow (if not already installed)

Run:
    python experiments/run_ablation.py
    mlflow ui
    python scripts/create_ablation_table.py
"""

import os
import json
import mlflow
from dotenv import load_dotenv

from src.config import ExperimentConfig, DENSE_CONFIG, HYBRID_CONFIG, AGENTIC_CONFIG
from src.data.corpus_download import load_arcd, load_wikipedia
from src.data.preprocessing import preprocess_corpus
from src.chunking.strategies import chunk_fixed_size
from src.agent.pipeline import ArRAGPipeline
from src.retrieval.qdrant_store import create_qdrant_store
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import RAGEvaluator

load_dotenv()

MLFLOW_EXPERIMENT_NAME = "ArRAG-Eval-Full"
NUM_TEST_SAMPLES = 20
CORPUS_SIZE = 500

CONFIGS_TO_TEST = [DENSE_CONFIG, HYBRID_CONFIG, AGENTIC_CONFIG]


def build_shared_index(base_config: ExperimentConfig):
    """Index the corpus once. All ablation configs query the same index —
    only the retrieval strategy at query time differs, so reindexing per
    config would just be paying 3x the embedding cost for nothing."""
    wiki = load_wikipedia(corpus_size=CORPUS_SIZE)
    wiki_cleaned = preprocess_corpus(wiki["text"])

    chunks = []
    for doc in wiki_cleaned:
        chunks.extend(
            chunk_fixed_size(doc, chunk_size=base_config.chunk_size, overlap=base_config.chunk_overlap)
        )

    vector_store = create_qdrant_store(
        collection_name=base_config.qdrant_collection_name,
        model_name=base_config.embedding_model,
        url=base_config.qdrant_url,
        api_key=base_config.qdrant_api_key or os.getenv("QDRANT_API_KEY"),
    )
    vector_store.add_documents(chunks, document_id="wikipedia_corpus")
    return vector_store


def run_one_config(config: ExperimentConfig, vector_store, test_questions, evaluator):
    print(f"\n{'=' * 60}")
    print(f"Running: {config.experiment_name}  "
          f"(retriever={config.retriever_type}, agentic={config.agentic})")
    print(f"{'=' * 60}")

    with mlflow.start_run(run_name=config.experiment_name):
        mlflow.log_params({k: str(v) for k, v in config.to_dict().items()})

        pipeline = ArRAGPipeline(config=config, vector_store=vector_store)
        results = pipeline.run_batch(test_questions)

        eval_results = []
        for i, result in enumerate(results):
            print(f"  Evaluating {i + 1}/{len(results)}...")
            context_texts = [c["text"] for c in result["retrieved_chunks"]]
            scores = evaluator.evaluate(
                question=result["question"],
                context=context_texts,
                answer=result["answer"],
            )
            eval_results.append(scores)

        aggregated = evaluator.aggregate_results(eval_results)
        mlflow.log_metrics({k: float(v) for k, v in aggregated.items()})

        serializable = [
            {name: mr.to_dict() for name, mr in sample.items()}
            for sample in eval_results
        ]
        mlflow.log_dict(serializable, "scores.json")

        outputs_path = f"ablation_{config.experiment_name}_outputs.json"
        with open(outputs_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        mlflow.log_artifact(outputs_path)

        print(
            f"✓ {config.experiment_name}: "
            f"precision={aggregated.get('context_precision', 0):.3f}  "
            f"faithfulness={aggregated.get('faithfulness', 0):.3f}  "
            f"relevance={aggregated.get('answer_relevance', 0):.3f}"
        )

        return aggregated


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Only used for shared indexing params (collection, embedding model,
    # chunk size) — all three ablation configs agree on these.
    base_config = DENSE_CONFIG.copy(deep=True)

    arcd = load_arcd()
    test_data = arcd["validation"][:NUM_TEST_SAMPLES]
    test_questions = test_data["question"]

    vector_store = build_shared_index(base_config)

    # Shared judge/evaluator across all three configs: same LLM, same
    # cache (.cache/judge_cache.db) — avoids re-paying for identical
    # judge decisions (e.g. dense and hybrid retrieving the same chunk
    # for the same question) across ablation runs.
    judge = LLMJudge(
        model=base_config.llm_model,
        host=base_config.ollama_host,
        language=base_config.language,
        use_cache=True,
    )
    evaluator = RAGEvaluator(judge=judge)

    summary = {}
    for raw_config in CONFIGS_TO_TEST:
        config = raw_config.copy(deep=True)
        config.num_test_samples = NUM_TEST_SAMPLES
        config.corpus_size = CORPUS_SIZE
        aggregated = run_one_config(config, vector_store, test_questions, evaluator)
        summary[config.experiment_name] = aggregated

    print(f"\n{'=' * 60}")
    print("ABLATION SUMMARY")
    print(f"{'=' * 60}")
    for name, scores in summary.items():
        print(
            f"{name:20s}  "
            f"precision={scores.get('context_precision', 0):.3f}  "
            f"faithfulness={scores.get('faithfulness', 0):.3f}  "
            f"relevance={scores.get('answer_relevance', 0):.3f}"
        )

    print("\n✓ All ablation runs logged to MLFlow. Run: mlflow ui")
    print("  Then build the table: python scripts/create_ablation_table.py")


if __name__ == "__main__":
    main()