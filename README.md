# ArRAG-Eval

ArRAG-Eval evaluates Arabic retrieval-augmented generation using Arabic Wikipedia as the retrieval corpus and questions from the ARCD validation split. The main ablation compares dense retrieval, hybrid retrieval, and hybrid retrieval with query decomposition.

The [project TODO](TODO.md) tracks benchmark fixes, implementation work, and follow-up experiments.

## How it works

1. The corpus loader fetches ARCD and a subset of Arabic Wikipedia. The experiment runners normalize the Arabic text and split it into overlapping fixed-size chunks. [Corpus loader](src/data/corpus_download.py) · [Preprocessing](src/data/preprocessing.py) · [Chunking](src/chunking/strategies.py)
2. The retrieval store embeds chunks with `intfloat/multilingual-e5-small` and stores them in Qdrant for dense search. BM25 ranks the same texts by keyword match in memory. Hybrid search combines both rankings with reciprocal rank fusion. [Retrieval store](src/retrieval/qdrant_store.py)
3. The LangGraph pipeline retrieves chunks and sends them to Ollama with the question. In agentic mode, it can break complex questions into sub-queries, retrieve for each, deduplicate chunks by ID, and assess context sufficiency. The graph proceeds directly from that check to generation. [Pipeline](src/agent/pipeline.py)
4. The scored runners use RAGEvaluator with the Ollama judge, then write per-question outputs and log aggregate metrics and artifacts to MLflow. [Metrics](src/evaluation/metrics.py) · [Judge](src/evaluation/judge.py)

The scored runners use the ARCD questions without using the reference answers. The pipeline's `evaluation_scores` field contains placeholder zeros. Use the evaluator output or the MLflow metrics for scored results. [Experiment presets](src/config.py)

## Run locally

Install dependencies from the repository root.

```powershell
python -m venv .myenv
.\.myenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run Qdrant at `http://localhost:6333`. In a separate terminal, start Ollama at `http://localhost:11434` if it is not running, then pull the default generation and judge model with `ollama pull qwen3:4b`.

```powershell
docker run -p 6333:6333 qdrant/qdrant:latest
```

The first run may download datasets and the embedding model. With the commands below run from the repository root, the corpus loader looks under the literal relative directory `./~/arrag-eval/data`.

| Command | What it does |
| --- | --- |
| `python -m experiments.run_baseline` | Runs hybrid retrieval for 20 questions and writes `baseline_outputs.json`. |
| `python -m experiments.run_baseline_metrics` | Scores the baseline and writes `detailed_scores.json`, reusing `baseline_metrics_outputs.json` if present. |
| `python -m experiments.run_ablation` | Builds or reuses one index for all three retrieval configurations, scores the same questions for each, and logs three MLflow runs. |
| `python -m scripts.create_ablation_table` | Reads the experiment configured in the script and writes `ablation_table.md`. |
| `python -m pytest -q -rs` | Runs the local integration smoke test when present. The test skips if Ollama or Qdrant is unreachable. |

## Reading the results

| Metric | What the implementation measures |
| --- | --- |
| Faithfulness | Fraction of generated claims the judge finds fully supported by retrieved context. |
| Answer relevance | Mean embedding similarity between the original question and questions generated from the answer. |
| Context precision | Mean precision across all retrieval cutoffs. The judge labels chunks by whether they contributed to the generated answer. |

The saved ablation in `ArRAG-Eval-Full` contains three completed runs with 20 questions per configuration and a logged corpus size of 500. Dense retrieval led on context precision and answer relevance, and agentic retrieval led on faithfulness in this run.

| Configuration | Context precision | Faithfulness | Answer relevance |
| --- | ---: | ---: | ---: |
| Dense | 0.374 | 0.580 | 0.812 |
| Hybrid | 0.362 | 0.516 | 0.799 |
| Agentic hybrid | 0.316 | 0.594 | 0.797 |

## Caveats for new comparisons

- The current ablation runner is set to 100 questions and logs to `ArRAG-Eval-Full 2`. The table generator and analysis notebook point at `ArRAG-Eval-Full`.
- The semantic metrics runner calls the fixed-size chunker and can reuse previously saved outputs. A semantic chunking comparison requires a separate run that actually uses semantic chunks. [Semantic metrics runner](experiments/run_semantic_metrics.py)
- Multiple runners reuse the `arrag-eval` collection and the same corpus ID. The store skips indexing when that ID already exists, so changing the chunking or corpus size can leave the previous index in use.
- When the store reloads an existing collection, BM25 uses Qdrant point IDs while dense search uses the chunk IDs stored in payloads. Hybrid fusion can then treat the same chunk as two independent results. The reload also reads a single page of up to 10,000 points.
- Agentic retrieval can supply more chunks than the other configurations because `top_k` applies per sub-query. The existing ablation compares the end-to-end configurations at their current settings.
