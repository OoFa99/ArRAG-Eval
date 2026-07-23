# ArRAG-Eval

A research-grade Arabic Retrieval-Augmented Generation evaluation repository.

This project builds and evaluates a hybrid Arabic RAG pipeline with:

- BM25 sparse retrieval
- dense semantic retrieval using SentenceTransformers
- hybrid retrieval fusion
- agentic query decomposition for multi-hop questions
- LLM-as-judge evaluation using RAGAS-style metrics
- MLFlow-based experiment tracking and ablation analysis

## Repository structure

- `src/`
  - `config.py` — Pydantic experiment configuration and preconfigured setups
  - `data/` — corpus loading and preprocessing utilities
  - `chunking/` — chunking strategies for retrieval corpus construction
  - `retrieval/` — Qdrant vector store and retrieval logic
  - `agent/` — agentic retrieval pipeline and decomposition flow
  - `evaluation/` — judge and metrics implementation
- `experiments/`
  - `run_baseline.py` — run a baseline retrieval + generation pipeline
  - `run_ablation.py` — run dense, hybrid, and agentic ablation experiments
  - `run_baseline_metrics.py` — baseline run with evaluation metrics
  - `run_semantic_metrics.py` — semantic retrieval experiment with metrics
- `scripts/`
  - `create_ablation_table.py` — generate a Markdown comparison table from MLFlow results
- `data_store/` — local data store (gitignored)
- `notebooks/` — analysis notebooks
- `tests/` — integration and module tests

## Requirements

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Recommended environment setup on Windows:

```powershell
python -m venv .myenv
.\.myenv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Prerequisites

- `Qdrant` running locally at `http://localhost:6333`
- `Ollama` server running locally at `http://localhost:11434`
- Local Ollama model available, e.g. `qwen3:4b`

Example Docker command for Qdrant:

```bash
docker run -p 6333:6333 qdrant/qdrant:latest
```

Example Ollama command:

```bash
ollama serve
ollama pull qwen3:4b
```

## Configuration

Default experiment settings are defined in `src/config.py`.

Key settings include:

- `retriever_type`: `dense`, `bm25`, or `hybrid`
- `agentic`: `True` or `False`
- `chunk_strategy`: `fixed`, `sentence`, or `semantic`
- `qdrant_url` and `qdrant_collection_name`
- `llm_model` and `ollama_host`

## Usage

### Run a baseline pipeline

```bash
python experiments/run_baseline.py
```

This script loads the corpus, preprocesses and chunks documents, indexes them in Qdrant, runs the RAG pipeline, and writes `baseline_outputs.json`.

### Run ablation experiments

```bash
python experiments/run_ablation.py
```

This runs three experiment configurations:

- `ablation-dense`
- `ablation-hybrid`
- `ablation-agentic`

All runs are logged to MLFlow under the experiment name `ArRAG-Eval-Full`.

### Generate an ablation results table

```bash
python scripts/create_ablation_table.py
```

This produces `ablation_table.md` from the latest MLFlow runs.

## Evaluation

Evaluation is performed by the LLM-based judge and RAGEvaluator in `src/evaluation/`.
Metrics tracked include:

- `context_precision`
- `faithfulness`
- `answer_relevance`
