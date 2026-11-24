# Why I built ArRAG-Eval

I wanted to understand a basic part of building an Arabic RAG system. When I ask a question in Arabic, does the system actually find the passage that can answer it?

That question became **ArRAG-Eval**, an open-source project for evaluating Arabic retrieval-augmented generation. A RAG system searches a collection of documents before asking a language model to answer. That gives the model evidence it can use, and gives me something concrete to inspect when an answer goes wrong.

For the first version, I split Arabic Wikipedia articles into passages and used questions from the Arabic Reading Comprehension Dataset (ARCD). Qdrant retrieves passages by embedding similarity, BM25 retrieves them by keyword match, and hybrid search combines their rankings. I also built a variant that breaks some questions into sub-queries. Ollama generates answers from the retrieved passages, and MLflow records the evaluation results.

The part I find most interesting is being able to follow a bad answer back through the pipeline. Was the evidence in the collection? Did search bring it back? Did the model use it? Each answer points to a different part of the system to improve.

In the next post, I’ll share what happened when I ran the first comparison on the same set of Arabic questions.

[ArRAG-Eval on GitHub](https://github.com/OoFa99/ArRAG-Eval)
