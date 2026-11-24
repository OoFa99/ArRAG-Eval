# How a restart changed hybrid search results

I gave ArRAG-Eval three passages and asked it to search for Cairo. After I reopened the Qdrant collection, hybrid search returned five ranked entries from those same three passages.

Hybrid search combines a dense ranking from Qdrant with a keyword ranking from BM25. When a passage appears in both lists, reciprocal rank fusion combines its contributions. That depends on both retrievers giving the passage the same ID.

The bug was in how I rebuilt BM25. Dense search returned the chunk ID saved in the payload. After a reload, BM25 returned Qdrant’s internal point ID for the same text. Fusion treated them as separate passages. In the test, the top passage’s score dropped from **0.03252 to 0.01639** after reopening.

I fixed the ID mismatch, made the rebuild follow every page from Qdrant, and gave BM25 ties a consistent order. Then I reran the same three-passage fixture with fixed embeddings. The top result scored **0.03279** both times, and all three passages came back in exactly the same order with the same scores.

That gave me a regression test I can run offline. It checks that the same passages keep their identity, rank, and score when the collection is reopened.
