# Reranking

**Status: not built yet.** This README is the plan, and there's no code to run here so far.

Use a second, more careful model to reorder the retrieved chunks, so the best ones reach the LLM.

## The problem

Embedding search is fast because it compares numbers computed separately for the question and for each chunk. The model never reads them together, so close calls can come out in the wrong order.

For example, on a small set of company documents, "Do I get any extra days off each month?" ranked a paragraph about vacation rollover first (0.669) and the right paragraph second (0.645). Only the top results reach the LLM, so a small ranking error can become a wrong answer.

## How it works

A reranker, usually a cross-encoder, reads the question and one chunk together and gives a single relevance score. That's more accurate but slower, and it can't be prepared in advance because it needs the question. So it runs as a second stage:

1. Embedding search finds a wide set of candidates, such as the top 20.
2. The reranker scores only those candidates.
3. The best few go to the LLM.

## What to build here

1. A pipeline that retrieves more candidates than it keeps.
2. A local cross-encoder through FastEmbed, which supports models such as `BAAI/bge-reranker-base`.
3. Logs that show each chunk's position before and after reranking, and the time taken.
4. A relevance cutoff based on the reranker's scores.

## Questions to answer

- Does reranking fix the example above?
- How many candidates are enough, and how much time does each one add?
- Do reranker scores separate relevant from irrelevant chunks better than cosine similarity?

## Further reading

- [Cross-encoder usage](https://sbert.net/docs/cross_encoder/usage/usage.html) in the Sentence Transformers docs
- [FastEmbed](https://github.com/qdrant/fastembed), the embedding library that also runs rerankers
