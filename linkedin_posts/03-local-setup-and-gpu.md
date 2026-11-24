# Getting Arabic RAG running on my laptop

When I reviewed the ArRAG-Eval setup, pytest failed before it reached a single RAG question. The installed versions of Transformers and Hugging Face Hub conflicted, and a separate Starlette version conflict affected MLflow.

I resolved the dependencies together for Python 3.12 and verified the result in a fresh Windows environment. I split the install into core, test, and notebook groups. On Windows, the core install dropped from 222 packages to 147, and it still has everything needed for the experiment runners.

I also wanted to know whether the project’s embedding model would actually run on my GTX 1650 Ti, which has 4 GB of VRAM. I created a separate CUDA environment and ran `multilingual-e5-small` on two Arabic texts. Both 384-dimensional output vectors were on the GPU. PyTorch reported 450 MiB of peak allocated GPU memory for that check.

The last piece was a startup command. On one run, it confirmed that Ollama had `qwen3:4b`, flagged that Qdrant was unreachable, and printed the paths where the loader expected the dataset caches. I could see what needed attention before spending time on indexing or generation.

[Local setup and GPU verification](https://github.com/OoFa99/ArRAG-Eval#run-locally)
