# RAG evaluation

**Status: not built yet.** This README is the plan, and there's no code to run here so far.

Measure how well a RAG system works with a script, instead of judging by eye from a few questions.

## The problem

Trying a few questions by hand is misleading. One good answer says little about the next one, and LLM output changes between runs. When you change a setting, you can't tell whether it helped overall or only on the question you happened to try.

## What to measure

RAG can fail in two places, so measure them separately:

| Part | Question it answers | Example measures |
|------|---------------------|------------------|
| Retrieval | Did the right chunk reach the LLM? | Hit rate (how often the right chunk is in the top results) and mean reciprocal rank, or MRR (how high it ranks) |
| Generation | Did the LLM use it properly? | Correct facts, claims backed by the chunks, and saying "I don't know" when the documents don't cover the question |

Retrieval measures need only the embedding model, so they run quickly. Answer checks need the LLM, and each question should run several times because the output varies.

## What to build here

1. A test set: questions with the expected chunk and answer, plus questions the documents can't answer.
2. A script that reports hit rate and MRR for retrieval alone.
3. Answer checks, starting with simple fact matching, such as requiring "23" in the answer, then trying an LLM as a judge.
4. Several runs per question, reported as a pass rate rather than a single pass or fail.
5. A comparison that changes one setting, such as `TOP_K` or the embedding model, and shows how the numbers move.

## Questions to answer

- How many test questions do you need before the numbers mean anything?
- How often does the model answer a question the documents don't cover?
- How far can you trust an LLM as a judge of another LLM's answers?

## Further reading

- [Ragas](https://docs.ragas.io/), a library of RAG evaluation measures
