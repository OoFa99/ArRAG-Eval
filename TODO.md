# ArRAG-Eval project TODO

The next milestone is a reproducible comparison of BM25, dense, and hybrid retrieval on Arabic questions with verified evidence. Use that benchmark to test semantic chunking, reranking, and agentic retrieval.

| Milestone | Deliverable | Depends on |
| --- | --- | --- |
| 1. Stabilize the stack and index | Clean setup, reliable tests, and reproducible retrieval after restart | — |
| 2. Build the benchmark | Versioned questions, verified evidence, and reference-based metrics | Stable document and chunk IDs |
| 3. Rerun the baseline | A paired, budget-controlled comparison with traceable MLflow runs | Milestones 1 and 2 |
| 4. Test retrieval and agent changes | Isolated experiments for chunking, reranking, decomposition, and retries | A reproducible baseline |
| 5. Improve maintainability | One experiment workflow with reproducible reports | Stable result and artifact formats |

## Audit snapshot

- In the local ARCD validation cache, the first 20 questions cover **3 article titles**. One title, accounting for 9 questions, matches an article among the first 500 cached Wikipedia rows. The first 100 questions cover **12 titles**, of which two match, accounting for 10 questions. This check matched normalized titles. Verify passage coverage separately because evidence may occur in another article or differ across snapshots.
- In the saved 20-question ablation, dense and hybrid returned 10 chunks per question. Agentic returned 10–29 chunks, averaging 12.5. Exact duplicate chunk text appeared in 8 hybrid results and 10 agentic results. These counts do not identify which mechanism produced each duplicate.
- The saved table reports the 20-question `ArRAG-Eval-Full` experiment. The current runner targets `ArRAG-Eval-Full 2` and 100 questions. Keep the saved table as historical evidence when rerunning the benchmark.
- The local integration test hit an import conflict between `transformers` and `huggingface-hub`. The environment also reported a Starlette version conflict with MLflow. Mocked graph checks exercised dense, hybrid, and agentic routing and reproduced a reported iteration without a second retrieval.

## 1. Stabilize the stack and index

### Environment and tests

- [x] Support Python 3.12 and lock a compatible dependency set. Verified a fresh Windows environment with `pip check` and pytest collection. [Dependency lock](uv.lock)
- [x] Split core, test, and notebook dependencies and verify a fresh Windows core install. Keep a pip-compatible export for core. [Dependency groups](pyproject.toml)
- [x] Verify optional GPU embeddings on Windows. The GTX 1650 Ti ran the default model with driver 621.14 and PyTorch 2.15.0 with CUDA 13.2. [GPU setup](README.md#optional-nvidia-gpu-setup-on-windows)
- [x] Add a startup check for service status, the required Ollama model, dataset cache paths, and Qdrant vector schema. [Startup command](scripts/check_startup.py)
- [ ] Pin the Qdrant image and record the Ollama model identity when preparing reproducible experiments.
- [x] Add CI for locked installs, dependency checks, linting, and offline tests. [CI workflow](.github/workflows/ci.yml)
- [x] Add a manual live smoke test with a small versioned fixture, strict HTTP status checks, and a disposable Qdrant collection. Verified locally with Ollama and Qdrant. [Smoke workflow](.github/workflows/smoke.yml)
- [x] Cover Recall@k with discoverable tests for repeated results, empty inputs and labels, and single and multiple cutoffs. [Recall tests](tests/test_recall.py)

Add regression tests alongside each correctness fix below.

### Index identity and consistency

- [x] Share canonical chunk IDs across dense and BM25 results, follow all Qdrant pages when rebuilding BM25, and keep tie order stable. Verified hybrid IDs, ranks, and scores across reopening. [Retrieval store](src/retrieval/qdrant_store.py) · [Restart tests](tests/test_qdrant_restart.py)
- [ ] Use stable point and chunk IDs instead of Python's process-dependent `hash()`. Store article ID, title, URL, revision, section, offsets, original text, and the searchable representation.
- [ ] Fingerprint corpus selection, preprocessing, chunking, embedding revision, and embedding input format. Use those fingerprints to select compatible collections so changed settings cannot silently reuse an older index.
- [ ] Make index builds resumable and detect incomplete batches. Verify expected counts and mark an index ready only after all writes succeed. Test retries, partial writes, and compatibility errors.
- [ ] Tighten Qdrant error handling. Distinguish a missing collection from network, authentication, or incompatible vector configuration errors, and surface partial BM25 loads explicitly.

**Done when** a clean environment runs the offline tests and the same fixed queries produce equivalent rankings across index restarts.

## 2. Build a coverage-verified benchmark

### Dataset and evidence

- [ ] Use one configurable cache directory. Support the existing `data_store/` cache and document how to migrate the literal relative `~/arrag-eval/data` path. [Corpus loader](src/data/corpus_download.py)
- [ ] Create a versioned manifest containing question IDs, article and passage references, accepted answers and offsets, dataset revisions, corpus IDs, and source licenses. Record actual document and chunk counts.
- [ ] Resolve ARCD titles and annotated passages to indexed articles and chunks. Use exact or normalized passage matching where available, and review unresolved cases caused by redirects, spelling variants, and article revisions.
- [ ] Create a controlled set with verified source passages and distractors. Also measure coverage in a broader Arabic Wikipedia corpus. Label unresolved questions so coverage gaps can be distinguished from retrieval misses.
- [ ] Sample across articles with a fixed seed. Keep development and final test articles disjoint. Report question and article counts for each split and preserve the exact evaluated IDs.
- [ ] Add labeled multi-hop and unanswerable subsets. Annotate each required evidence passage for multi-hop questions. Include missing evidence, ambiguous questions, and false premises in the unanswerable subset.
- [ ] Add evaluation slices for names, dates, numbers, spelling variants, paraphrases, and code-switching. Add dialectal questions or non-Wikipedia domains when those become explicit product goals. [ARCD paper](https://aclanthology.org/W19-4612/)

### Metrics and judging

- [ ] Add separate retrieval metrics from evidence labels. Start with document and passage Recall@k and MRR. Add nDCG where graded labels exist, and measure whether all required evidence was recovered for multi-hop questions.
- [ ] Score short answers with Arabic-normalized exact match and token F1 over accepted variants. Use an Arabic human-reviewed sample for longer or ambiguous answers. Track correctness, faithfulness, relevance, citation quality, and abstention as distinct outcomes.
- [ ] Rename and version the local `context_precision` metric or implement the documented Ragas formula. The local function averages precision at all ranks. Use question-to-evidence labels for retrieval, and keep judgments about which chunks contributed to the generated answer as a separate context-use signal. [Local metrics](src/evaluation/metrics.py) · [Ragas formula](https://docs.ragas.io/en/v0.2.0/concepts/metrics/available_metrics/context_precision/)
- [ ] Calibrate the judge against a blinded human-rated Arabic sample. Measure agreement and review errors involving unsupported claims, abstentions, and valid paraphrases. Compare an independent judge model where practical.
- [ ] Add explicit statuses for timeouts, malformed responses, abstentions, and scoring failures. Report valid denominators and failure rates. Replace the graph's placeholder `evaluation_scores` with the actual evaluation result or remove the field through a versioned output change.
- [ ] Key judge results by a hash of the full inputs, model and prompt versions, language, and decoding settings. Add regression cases for long inputs with a shared prefix. [Judge cache](src/evaluation/judge.py)

**Done when** every question in the controlled test set has verifiable evidence labels or a deliberate unanswerable label, and retrieval and answer scores can be reproduced from saved outputs.

## 3. Rerun the baseline

- [ ] Apply the multilingual E5 model's `query:` and `passage:` formats and rebuild the index. Measure chunks with the embedding tokenizer and enforce its 512-token limit, leaving space for prefixes and special tokens. [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small)
- [ ] Run BM25-only, dense, and hybrid on identical question IDs, corpus, model settings, and final context budgets. Add no-retrieval and gold-evidence runs to locate the quality gap between retrieval and generation.
- [ ] Compare normalization for corpus and query tokens, and exclude candidates with no lexical match before RRF. Deduplicate by canonical ID and detect exact duplicates and overlapping passages. Record candidate depths, final chunk counts, and prompt token counts.
- [ ] Log a shared comparison ID in MLflow. Save Git revision, dataset and index fingerprints, exact question IDs, effective config, model revisions, decoding settings, prompt and metric versions, latency, model calls, tokens, and failure counts. Exclude API keys and other credentials.
- [ ] Make the table and notebook accept a tracking URI, experiment, and comparison ID. Select completed runs with matching inputs, include source run IDs, and remove the notebook's absolute local tracking path. [Ablation runner](experiments/run_ablation.py) · [Table generator](scripts/create_ablation_table.py) · [Analysis notebook](notebooks/ablation_analysis.ipynb)
- [ ] Report per-question paired deltas, win/tie/loss counts, and confidence intervals. Resample by article to account for clustered questions. Repeat stochastic runs and report latency percentiles alongside quality.

**Done when** one manifest and comparison ID reproduce the expected run cohort and report, including metrics, input counts, and resource use.

## 4. Test improvements individually

### Arabic retrieval and chunking

- [ ] Wire fixed, sentence, and semantic chunking through a single config-driven selector. Move the useful hybrid chunking prototype into the tested path. Fix invalid overlaps, dropped short passages, and over-limit single sentences. Make structure-aware splitting preserve paragraph and section boundaries. [Chunking strategies](src/chunking/strategies.py) · [Semantic runner](experiments/run_semantic_metrics.py)
- [ ] Benchmark Arabic normalization, punctuation handling, clitic segmentation, light stemming, and character n-grams. Evaluate CAMeL Tools where it fits. Keep original passage text and measure the effects on names, digits, and code-switched queries.
- [ ] Sweep chunk size, overlap, candidate depth, final `k`, and RRF parameters on the development split. Evaluate a stronger multilingual encoder or Arabic-capable reranker only after the baseline is stable. Compare relevance gains against latency and memory.
- [ ] Test deduplication of overlapping evidence and cross-sub-query reranking. Maintain distinct passages required for multi-hop answers and evaluate evidence coverage after packing.

### Agentic retrieval and generation

- [ ] Fix Ollama host and model propagation for decomposition, sufficiency, generation, and judging. Check these settings at each call site or inject one configured client. [Decomposer](src/agent/decomposer.py) · [Pipeline](src/agent/pipeline.py)
- [ ] Evaluate decomposition independently. Use labeled Arabic queries to test the complexity heuristic, replace broad substring checks such as any occurrence of `و`, and cap, deduplicate, and validate sub-queries. Compare retaining the original query with decomposition alone.
- [ ] Treat decomposition-only and retry-enabled retrieval as separate configurations. Wire a bounded conditional retry that queries for missing evidence, merges unique results, stops when evidence stops improving, and records actual retrieval calls.
- [ ] Enforce a final prompt token budget. Reserve prompt and answer space, prioritize complementary evidence, and log the chunks supplied to the model. Compare configurations under equal budgets and report native-budget results separately.
- [ ] Validate typed model responses for sub-queries, sufficiency, and judging. Bound retries across layers, distinguish transient and permanent errors, and keep failed responses out of normal scoring.
- [ ] Use an explicit Arabic answer policy with source citations and structured abstention. Validate cited chunk IDs and evaluate whether citations support claims. Treat retrieved text as quoted source material and test that instructions embedded in documents do not control generation.

**Done when** each experiment changes a single component and reports its effects on evidence recall, answer quality, tokens, model calls, and latency.

## 5. Improve maintainability

- [ ] Consolidate duplicate experiment setup into one entry point for loading, preprocessing, indexing, generation, scoring, and reporting. Keep the scripts importable without starting work. Document supported commands in the README.
- [ ] Make config fields drive behavior or remove inactive options such as `chroma` and `use_judge`. Migrate deprecated Pydantic APIs, validate overlap and RRF settings, and add tests for effective configuration.
- [ ] Replace filename-only result caches with run fingerprints. Write resumable per-question outputs keyed by question ID, and version the output schema. Keep comparison manifests and small fixtures trackable by narrowing blanket ignore patterns.
- [ ] Profile embedding and judge calls, reuse initialized models, and batch independent work where practical. Keep concurrency bounded for the local Ollama instance and make timing visible per stage.
- [ ] Retire superseded prototypes and outdated comments after their tested replacements land. Add a short workflow for benchmark contributors. If the project expands into another domain, first define a corpus and annotation interface with provenance, licensing, and a domain-specific benchmark.

**Done when** a contributor can install the project, run a small reproducible comparison, inspect its saved evidence, and regenerate its report using documented commands.
