"""
Runs the semantic-chunking pipeline and scores every result with the LLM-judge-based RAGEvaluator.

This is a copy of experiments/run_baseline_metrics.py, but it uses SEMANTIC_CONFIG
from src.config so the pipeline uses semantic chunking instead of fixed-size chunks.

Run:
    python experiments/run_semantic_metrics.py
"""

import os
import json
import mlflow
from dotenv import load_dotenv

from src.config import SEMANTIC_CONFIG
from src.data.corpus_download import load_arcd, load_wikipedia
from src.data.preprocessing import preprocess_corpus
from src.chunking.strategies import chunk_fixed_size
from src.agent.pipeline import ArRAGPipeline
from src.retrieval.qdrant_store import create_qdrant_store
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import RAGEvaluator

load_dotenv()

MLFLOW_EXPERIMENT_NAME = "ArRAG-Eval_semantic_metrics"
SEMANTIC_OUTPUTS_PATH = "semantic_metrics_outputs.json"
DETAILED_SCORES_PATH = "semantic_detailed_scores.json"


def get_pipeline_results(config):
    """Reuse semantic_metrics_outputs.json if present, otherwise run the pipeline fresh."""
    if os.path.exists(SEMANTIC_OUTPUTS_PATH):
        print(f"Found existing {SEMANTIC_OUTPUTS_PATH}, reusing it "
              f"(delete the file to force a fresh pipeline run).")
        with open(SEMANTIC_OUTPUTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    print("No cached outputs found — running the pipeline from scratch.")
    arcd = load_arcd()
    wiki = load_wikipedia(corpus_size=config.corpus_size)
    wiki_cleaned = preprocess_corpus(wiki["text"])

    chunks = []
    for doc in wiki_cleaned:
        chunks.extend(
            chunk_fixed_size(doc, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        )

    vector_store = create_qdrant_store(
        collection_name=config.qdrant_collection_name,
        model_name=config.embedding_model,
        url=config.qdrant_url,
        api_key=config.qdrant_api_key or os.getenv("QDRANT_API_KEY"),
    )
    vector_store.add_documents(chunks, document_id="wikipedia_corpus")

    pipeline = ArRAGPipeline(config=config, vector_store=vector_store)
    test_data = arcd["validation"][: config.num_test_samples]
    results = pipeline.run_batch(test_data["question"])

    with open(SEMANTIC_OUTPUTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def main():
    config = SEMANTIC_CONFIG.copy(deep=True)
    config.num_test_samples = 20

    results = get_pipeline_results(config)

    judge = LLMJudge(
        model=config.llm_model,
        host=config.ollama_host,
        language=config.language,
        use_cache=True,
    )
    evaluator = RAGEvaluator(judge=judge)

    eval_results = []
    for i, result in enumerate(results):
        print(f"Evaluating {i + 1}/{len(results)}...")
        context_texts = [c["text"] for c in result["retrieved_chunks"]]
        scores = evaluator.evaluate(
            question=result["question"],
            context=context_texts,
            answer=result["answer"],
        )
        eval_results.append(scores)

    aggregated = evaluator.aggregate_results(eval_results)

    print("\nFinal Scores:")
    print(f"  Faithfulness:      {aggregated.get('faithfulness', 0):.3f}")
    print(f"  Answer Relevance:  {aggregated.get('answer_relevance', 0):.3f}")
    print(f"  Context Precision: {aggregated.get('context_precision', 0):.3f}")

    serializable_eval_results = [
        {metric_name: metric_result.to_dict() for metric_name, metric_result in sample.items()}
        for sample in eval_results
    ]
    with open(DETAILED_SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_eval_results, f, indent=2, ensure_ascii=False)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="week2-semantic-with-eval"):
        mlflow.log_params({k: str(v) for k, v in config.to_dict().items()})
        mlflow.log_metrics({k: float(v) for k, v in aggregated.items()})
        mlflow.log_dict(serializable_eval_results, "semantic_detailed_scores.json")

    print(f"\n✓ Complete. Scores logged to MLFlow and saved to {DETAILED_SCORES_PATH}.")


if __name__ == "__main__":
    main()
