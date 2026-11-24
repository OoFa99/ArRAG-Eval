# What the first Arabic retrieval comparison showed

In my first comparison of ArRAG-Eval’s retrieval modes, dense retrieval scored higher on context precision and answer relevance. Hybrid retrieval with query decomposition scored slightly higher on faithfulness.

I ran three configurations on the same 20 Arabic questions. Here are the scores in order of context precision, faithfulness, and answer relevance.

- Dense retrieval — **0.374 / 0.580 / 0.812**
- Hybrid retrieval — **0.362 / 0.516 / 0.799**
- Hybrid with query decomposition — **0.316 / 0.594 / 0.797**

Faithfulness here measures how much of the generated answer an LLM judge found supported by the retrieved text. The decomposition variant also returned more passages for some questions, so it sometimes gave the generator more context to work with.

The bigger finding came when I looked at the data. Those 20 questions came from just three source article titles. Only one matched a title in the first 500 cached Arabic Wikipedia rows. Other articles may still contain relevant evidence, so I need a passage-level check to know what was actually available.

I’m treating these scores as observations from a small run. Before drawing conclusions about which retriever works best, I want to verify that the collection contains the evidence, use a more varied set of questions, and give every configuration the same context budget.

[Saved ablation results](https://github.com/OoFa99/ArRAG-Eval/blob/main/ablation_table.md)
