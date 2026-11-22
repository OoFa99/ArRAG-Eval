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

Use Python 3.12 and run these commands from the repository root. The default installation includes the experiment pipeline. Tests and notebooks are opt-in. [uv dependency groups](https://docs.astral.sh/uv/concepts/dependencies/#dependency-groups)

| What you need | Setup |
| --- | --- |
| Run experiments | `uv sync --locked` |
| Run experiments and tests | `uv sync --locked --group test` |
| Run experiments and notebooks | `uv sync --locked --group notebook` |
| Use both optional groups | `uv sync --locked --group test --group notebook` |

Run the offline tests with `uv run --locked --group test python -m pytest -q -m "not integration"` and lint with `uv run --locked --group test ruff check`. GitHub Actions runs these checks on pushes and pull requests. Open the notebook with `uv run --locked --group notebook jupyter lab`.

If you use pip, install the exported core requirements in a fresh environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

To update dependencies, edit `pyproject.toml`, refresh the shared lock, and regenerate the core pip export. Use `uv lock --upgrade` when refreshing every pin. [uv export reference](https://docs.astral.sh/uv/reference/cli/#uv-export)

```powershell
uv lock
uv export --locked --no-default-groups --no-emit-project --no-hashes --output-file requirements.txt
```

The default Windows setup was verified with CPU PyTorch. The [optional GPU setup](#optional-nvidia-gpu-setup-on-windows) uses a separate environment for CUDA embeddings.

Run Qdrant at `http://localhost:6333`. In a separate terminal, start Ollama at `http://localhost:11434` if it is not running, then pull the default generation and judge model with `ollama pull qwen3:4b`.

```powershell
docker run -p 6333:6333 qdrant/qdrant:latest
```

The first run may download datasets and the embedding model. With the commands below run from the repository root, the corpus loader looks under the literal relative directory `./~/arrag-eval/data`.

Run `uv run --locked python -m scripts.check_startup` before starting an experiment. With an activated pip environment, use `python -m scripts.check_startup`.

The check uses the baseline settings and reads the Ollama model list, the Qdrant collection schema, and local dataset cache metadata. A missing cache or collection is a first-run warning. An unreachable service, a missing Ollama model, or an incompatible collection returns exit code 1. For the default E5 model, the collection should have a single unnamed 384-dimensional dense vector with Cosine distance. Verifying which model produced existing vectors requires separate index metadata.

| Command | What it does |
| --- | --- |
| `python -m experiments.run_baseline` | Runs hybrid retrieval for 20 questions and writes `baseline_outputs.json`. |
| `python -m experiments.run_baseline_metrics` | Scores the baseline and writes `detailed_scores.json`, reusing `baseline_metrics_outputs.json` if present. |
| `python -m experiments.run_ablation` | Builds or reuses one index for all three retrieval configurations, scores the same questions for each, and logs three MLflow runs. |
| `python -m scripts.create_ablation_table` | Reads the experiment configured in the script and writes `ablation_table.md`. |

### Live pipeline smoke test

Start Qdrant and Ollama, then run the smoke test explicitly. The test requires successful HTTP responses from both services and uses a small Arabic corpus from the repository. It creates a temporary Qdrant collection and removes it when the test finishes.

```powershell
uv run --locked --group test python -m pytest -q -m integration --run-integration
```

On GitHub, run **Live pipeline smoke test** manually from the **Actions** tab. That workflow starts both services and pulls the smaller `qwen3:0.6b` model for the CPU runner.

### Optional NVIDIA GPU setup on Windows

Verified on a GeForce GTX 1650 Ti with 4 GB of VRAM, NVIDIA driver 621.14, and Python 3.12.3. The default embedding model, `intfloat/multilingual-e5-small`, encoded two Arabic texts as 384-dimensional vectors on `cuda:0`. PyTorch reported 450 MiB of peak allocated GPU memory for this check.

Check your card and driver with `nvidia-smi`. The commands below use Python 3.12 and the PyTorch 2.15.0 wheel built with CUDA 13.2. The CUDA version shown by `nvidia-smi` is the driver's supported version. `torch.version.cuda` reports the version used to build the wheel. [PyTorch Windows CUDA wheels](https://download.pytorch.org/whl/cu132/torch/) · [NVIDIA CUDA driver compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

```powershell
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu132 --no-deps "torch==2.15.0+cu132"
uv pip install --python .venv-gpu/Scripts/python.exe -r requirements.txt
uv pip check --python .venv-gpu/Scripts/python.exe
```

Test the same embedding model used by the default pipeline. The first run may download its weights.

```powershell
@'
import torch
from sentence_transformers import SentenceTransformer

assert torch.cuda.is_available(), f"CUDA unavailable in PyTorch {torch.__version__}"
model = SentenceTransformer("intfloat/multilingual-e5-small")
embedding = model.encode(["query: Cairo"], convert_to_tensor=True)
assert embedding.device.type == "cuda", f"Embedding is on {embedding.device}"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(embedding.shape, embedding.device)
'@ | .\.venv-gpu\Scripts\python.exe -
```

Use the GPU environment's Python for commands that should use this PyTorch build.

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
