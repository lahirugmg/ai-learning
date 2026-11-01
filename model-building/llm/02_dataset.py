"""
Step 2 — Dataset: Prepare training examples for the model.

A language model learns to predict the next word. We slide a window of
`context_length` words across the text and ask: given these words, what comes
next?

For context_length=4 and the text "the cat sat on a mat":
    Input: [the, cat, sat, on]   → Target: a
    Input: [cat, sat, on, a]     → Target: mat

Run:
    uv run python 02_dataset.py
"""

import json

import torch
from torch.utils.data import Dataset


class NextWordDataset(Dataset):
    """A dataset of (context, next_word) pairs.

    Each item is:
        x: tensor of `context_length` token ids  (the input)
        y: a single token id                      (the word to predict)
    """

    def __init__(self, tokens: list[int], context_length: int = 4):
        self.context_length = context_length
        self.examples = []
        # Slide a window across the tokens
        for i in range(len(tokens) - context_length):
            x = tokens[i : i + context_length]
            y = tokens[i + context_length]
            self.examples.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


if __name__ == "__main__":
    with open("data/tokens.json") as f:
        tokens = json.load(f)
    with open("data/vocab.json") as f:
        word2idx = json.load(f)

    idx2word = {i: w for w, i in word2idx.items()}

    dataset = NextWordDataset(tokens, context_length=4)
    print(f"Total training examples: {len(dataset)}")
    print()

    # Show the first 5 examples
    print("First 5 training examples:")
    print(f"  {'Input (4 words)':<35} → {'Target'}")
    print(f"  {'─' * 35}   {'─' * 10}")
    for i in range(5):
        x, y = dataset[i]
        words_in = " ".join(idx2word[t.item()] for t in x)
        word_out = idx2word[y.item()]
        print(f"  {words_in:<35} → {word_out}")

