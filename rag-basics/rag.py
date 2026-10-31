"""Minimal RAG with no framework: chunk, embed, retrieve, generate.

Usage: uv run rag.py "How many vacation days do new hires get?"
"""

import sys
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from openai import OpenAI

LLM_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DOCS_DIR = Path(__file__).parent / "docs"
TOP_K = 3

llm = OpenAI(base_url=LLM_URL, api_key="not-needed")
embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(Path.home() / ".cache" / "fastembed"))


def load_chunks():
    """Split each document into paragraphs, keeping the file name for citations."""
    chunks = []
    # Searches subfolders too, so it also reads docs/private/, which git ignores.
    for path in sorted(DOCS_DIR.rglob("*.md")):
        for paragraph in path.read_text().split("\n\n"):
            paragraph = paragraph.strip()
            # A heading alone says little, and would use up one of the TOP_K slots.
            if paragraph and not paragraph.startswith("#"):
                chunks.append({"source": path.name, "text": paragraph})
    return chunks


def retrieve(question, chunks, vectors):
    """Return the TOP_K chunks closest in meaning to the question."""
    query = next(iter(embedder.query_embed(question)))
    scores = vectors @ query  # The vectors are normalized, so this is cosine similarity.
    best = np.argsort(scores)[::-1][:TOP_K]
    return [(chunks[i], float(scores[i])) for i in best]


def generate(question, hits):
    """Ask the LLM to answer from the retrieved chunks only."""
    context = "\n\n".join(f"[{chunk['source']}]\n{chunk['text']}" for chunk, _ in hits)
    system = (
        "Answer using only the context below and cite the source file in brackets. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}"
    )
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=500,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    question = sys.argv[1]
    chunks = load_chunks()
    vectors = np.array(list(embedder.passage_embed([c["text"] for c in chunks])))
    hits = retrieve(question, chunks, vectors)

    print("Retrieved:")
    for chunk, score in hits:
        print(f"  {score:.3f}  {chunk['source']}: {chunk['text'][:60]}...")
    print(f"\nAnswer:\n{generate(question, hits)}")
