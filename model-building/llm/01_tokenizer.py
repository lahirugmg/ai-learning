"""
Step 1 — Tokenizer: Turn text into numbers the model can read.

A neural network only understands numbers, not letters. A tokenizer maps each
unique word to an integer. Real LLMs use subword tokenizers (BPE), but
word-level is easier to understand and enough for our tiny dataset.

Run:
    uv run python 01_tokenizer.py
"""

import json


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Build word-to-index and index-to-word mappings.

    Steps:
      1. Split the text into words (lowercased).
      2. Collect unique words and sort them so the mapping is deterministic.
      3. Assign each word an integer starting from 0.

    Returns:
        word2idx: {"the": 0, "cat": 1, ...}
        idx2word: {0: "the", 1: "cat", ...}
    """
    words = text.lower().split()
    vocab = sorted(set(words))  # unique words, sorted alphabetically
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}
    return word2idx, idx2word


def tokenize(text: str, word2idx: dict[str, int]) -> list[int]:
    """Convert a string of words into a list of integers.

    Example:
        "the cat sat" → [42, 7, 35]  (the actual numbers depend on the vocab)
    """
    return [word2idx[w] for w in text.lower().split()]


def detokenize(tokens: list[int], idx2word: dict[int, str]) -> str:
    """Convert a list of integers back into a string of words."""
    return " ".join(idx2word[t] for t in tokens)


if __name__ == "__main__":
    # Load the training text
    with open("data/tiny-stories.txt") as f:
        text = f.read()

    word2idx, idx2word = build_vocab(text)
    tokens = tokenize(text, word2idx)

    print(f"Text has {len(text.split())} words")
    print(f"Vocabulary size: {len(word2idx)} unique words")
    print()

    # Show a few mappings
    print("Sample word → number mappings:")
    for word in ["the", "cat", "sat", "on", "a"]:
        print(f"  '{word}' → {word2idx[word]}")
    print()

    # Show tokenization of a sentence
    sentence = "the cat sat on a mat"
    encoded = tokenize(sentence, word2idx)
    decoded = detokenize(encoded, idx2word)
    print(f"Tokenize:   '{sentence}'")
    print(f"  → numbers: {encoded}")
    print(f"  → back:    '{decoded}'")

    # Save for the next step
    with open("data/vocab.json", "w") as f:
        json.dump(word2idx, f)
    with open("data/tokens.json", "w") as f:
        json.dump(tokens, f)
    print(f"\nSaved vocab ({len(word2idx)} words) and tokens ({len(tokens)} ints) to data/")

