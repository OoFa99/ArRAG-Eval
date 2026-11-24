# What I test before trusting the pipeline

Here’s a small retrieval test I added to ArRAG-Eval. If the relevant documents are A and B, and search returns A, A, B, what should Recall@2 be?

It should be **0.5**. The first two results contain one distinct relevant document, so search found one of the two documents we needed. That example checks both duplicate handling and whether the cutoff is applied at the right point.

I added 10 Recall@k cases, including empty results, missing relevance labels, and cutoffs beyond the end of the list. They now run with the startup-check tests. The GitHub Actions workflow installs the locked dependencies, checks for conflicts, runs Ruff, and runs the offline suite on pushes and pull requests.

The separate live test uses a tiny Arabic corpus checked into the repository. It starts with a disposable Qdrant collection, indexes the fixture, retrieves a passage, and asks Ollama a question about it. The test cleans up the collection afterward.

Locally, all 20 offline tests passed, and the live test passed against Ollama and Qdrant. The two layers answer different questions. The unit tests check rules I can express precisely. The live test checks whether the pieces can actually work together.

[Test suite](https://github.com/OoFa99/ArRAG-Eval/tree/main/tests)
