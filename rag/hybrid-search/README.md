# Hybrid search

**Status: not built yet.** This README is the plan, and there's no code to run here so far.

Combine keyword search with meaning search, so retrieval catches exact terms as well as differently worded questions.

## The problem

Embedding search matches by meaning, so a question about a stolen computer can find a passage about laptops. But it can miss exact terms such as product codes, error numbers and unusual names, because an embedding captures the general sense of the text rather than each word. Keyword search has the opposite strengths:

| Question | Meaning search | Keyword search |
|----------|----------------|----------------|
| "What if my computer gets stolen?", when the document says "laptops" | Usually finds it | Misses it unless other words match |
| "What does error E-4417 mean?" | Can miss it | Finds it |

## How it works

1. Run both searches on the same chunks. Keyword search usually means BM25, which scores chunks by the words they share with the question and gives rare words more weight.
2. Combine the two ranked lists. Their scores aren't comparable, so a common method is reciprocal rank fusion (RRF), which uses each chunk's position in each list rather than its score.
3. Keep the best few chunks and send them to the LLM as usual.

## What to build here

1. Sample documents with exact terms that embedding search struggles with, such as product codes and error numbers.
2. BM25 written by hand, so you can see how the scoring works.
3. Reciprocal rank fusion in a few lines of Python.
4. Logs that show all three rankings side by side: meaning, keyword and combined.
5. Test questions that each search gets wrong on its own.

## Questions to answer

- Which questions does each search win on, and does combining them ever make things worse?
- How much does RRF lose by ignoring the scores and using only positions?

## Further reading

- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) on Wikipedia
- [The reciprocal rank fusion paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (Cormack et al., 2009)
- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval), on combining embeddings with BM25
