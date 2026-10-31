# RAG with a vector database

**Status: not built yet.** This README is the plan, and there's no code to run here so far.

Save chunks and embeddings on disk, so they don't have to be recomputed on every run.

## The problem

A simple RAG script can keep its embeddings in a NumPy array in memory and rebuild them each time it starts. That's fine for a few documents: 133 short chunks took 0.46 seconds on a Mac mini's CPU. At that rate, 10,000 chunks would take very roughly 35 seconds on every run, and changing one document would mean redoing all of them.

## What a vector database adds

| Feature | What it gives you |
|---------|-------------------|
| Saving to disk | Embeddings survive between runs |
| Updates | Add, change or delete one document without embedding the rest again |
| Metadata filters | Search only within, say, one source file or date range |
| Fast search at scale | Approximate indexes such as HNSW find close matches without comparing every vector, trading a little accuracy for speed |

It doesn't improve answers by itself. With the same chunks and embeddings, an exact search returns the same results as the NumPy array, and an approximate search can only match or fall short of that.

Saving also brings a new risk: stale embeddings. You have to update them when a document changes, and rebuild all of them when you change the embedding model, because vectors from different models can't be compared.

## What to build here

1. A version with no database that saves to files with `np.save` and JSON, to see what's still missing.
2. Change detection that hashes each document and embeds only the ones that changed.
3. A move to a vector database, comparing options such as Chroma, Qdrant and sqlite-vec.
4. Logs that show what each run loads, embeds again and skips.
5. A timing comparison of exact and approximate search as the collection grows.

## Questions to answer

- At what size does saving start to pay off, and when does a NumPy array actually get too slow?
- How do you tell that saved embeddings are out of date?
- How much accuracy does approximate search lose?

## Further reading

- [The HNSW paper](https://arxiv.org/abs/1603.09320) (Malkov and Yashunin, 2016)
- [Chroma](https://docs.trychroma.com/), [Qdrant](https://qdrant.tech/documentation/) and [sqlite-vec](https://github.com/asg017/sqlite-vec)
