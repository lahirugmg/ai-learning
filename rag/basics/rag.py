"""Minimal RAG with no framework: chunk, embed, retrieve, generate.

Prints what happens at each step, so you can follow a question through the pipeline.

Usage: uv run rag.py "How many vacation days do new hires get?"
"""

import sys
import textwrap
import time
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

LLM_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DOCS_DIR = Path(__file__).parent / "docs"
TOP_K = 3
MAX_SCORES_SHOWN = 15

llm = OpenAI(base_url=LLM_URL, api_key="not-needed")
embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(Path.home() / ".cache" / "fastembed"))


def log_step(number, title):
    print(f"\n[{number}/6] {title}")


def format_vector(vector):
    """Show the first few numbers of an embedding, how many it has, and its length."""
    numbers = ", ".join(f"{n:.3f}" for n in vector[:5])
    return f"[{numbers}, ...] {len(vector)} numbers, length {np.linalg.norm(vector):.3f}"


def load_chunks():
    """Split each document into paragraphs, keeping the file name for citations."""
    log_step(1, "CHUNK: split each document into paragraphs")
    chunks = []
    # Searches subfolders too, so it also reads docs/private/, which git ignores.
    for path in sorted(DOCS_DIR.rglob("*.md")):
        paragraphs = [p.strip() for p in path.read_text().split("\n\n")]
        # A heading alone says little, and would use up one of the TOP_K slots.
        kept = [p for p in paragraphs if p and not p.startswith("#")]
        chunks += [{"source": path.name, "text": p} for p in kept]
        print(f"  {path.name}: {len(kept)} chunk(s), longest {max(map(len, kept), default=0)} characters")
    print(f"  Total: {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks):
    """Turn every chunk into a vector, stored together in one array."""
    log_step(2, "EMBED CHUNKS: turn each chunk into numbers that represent its meaning")
    start = time.perf_counter()
    vectors = np.array(list(embedder.passage_embed([c["text"] for c in chunks])))
    print(f"  Model: {EMBED_MODEL}, running inside this script")
    print(f"  Result: {vectors.shape[0]} chunks x {vectors.shape[1]} numbers, {vectors.nbytes / 1024:.0f} KB, {time.perf_counter() - start:.2f}s")
    print(f"  First chunk: {format_vector(vectors[0])}")
    return vectors


def retrieve(question, chunks, vectors):
    """Return the TOP_K chunks closest in meaning to the question."""
    log_step(3, "EMBED QUESTION: turn the question into numbers with the same model")
    query = next(iter(embedder.query_embed(question)))
    print(f"  Question: {question}")
    print(f"  Vector: {format_vector(query)}")

    log_step(4, f"RETRIEVE: score {len(chunks)} chunks against the question, keep the best {TOP_K}")
    scores = vectors @ query  # The vectors are normalized, so this is cosine similarity.
    ranked = np.argsort(scores)[::-1]
    print("  Score is cosine similarity: higher means closer in meaning.")
    shown = max(MAX_SCORES_SHOWN, TOP_K + 1)
    for rank, i in enumerate(ranked[:shown]):
        if rank == TOP_K:
            print(f"  ---- TOP_K = {TOP_K}: chunks below this line are dropped ----")
        print(f"  {scores[i]:.3f}  {chunks[i]['source']}: {chunks[i]['text'][:60]}...".replace("\n", " "))
    if len(ranked) > shown:
        print(f"  ... and {len(ranked) - shown} lower scores")
    return [(chunks[i], float(scores[i])) for i in ranked[:TOP_K]]


def generate(question, hits):
    """Ask the LLM to answer from the retrieved chunks only."""
    log_step(5, "BUILD PROMPT: this is everything the LLM will see")
    context = "\n\n".join(f"[{chunk['source']}]\n{chunk['text']}" for chunk, _ in hits)
    system = (
        "Answer using only the context below and cite the source file in brackets. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    for message in messages:
        print(f"  {message['role']}:")
        print(textwrap.indent(message["content"], "    "))

    log_step(6, "GENERATE: send the prompt to the LLM and wait for its answer")
    settings = dict(
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=500,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    print(f"  Request: POST {LLM_URL}/chat/completions")
    print(f"  Settings: {settings}")
    start = time.perf_counter()
    response = llm.chat.completions.create(messages=messages, **settings)
    choice = response.choices[0]
    print(f"  Response: {time.perf_counter() - start:.1f}s, finish_reason: {choice.finish_reason}")
    if response.usage:
        print(f"  Tokens: {response.usage.prompt_tokens} in the prompt, {response.usage.completion_tokens} generated")
    # Only there when thinking is turned on (experiment 7 in the README).
    thinking = getattr(choice.message, "reasoning", None)
    if thinking:
        print("  Thinking:")
        print(textwrap.indent(thinking.strip(), "    "))
    return choice.message.content.strip()


if __name__ == "__main__":
    question = sys.argv[1]
    chunks = load_chunks()
    vectors = embed_chunks(chunks)
    hits = retrieve(question, chunks, vectors)
    print(f"\nAnswer:\n{generate(question, hits)}")
