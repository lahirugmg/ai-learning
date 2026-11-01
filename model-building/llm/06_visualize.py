"""
Step 6 — Visualize: Plot the training loss and attention weights.

Run:
    uv run python 06_visualize.py
"""

import json

import matplotlib.pyplot as plt
import torch

import importlib

_model_mod = importlib.import_module("03_model")
TinyTransformer = _model_mod.TinyTransformer


def plot_loss(losses: list[float]):
    """Plot training loss over epochs."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, linewidth=2, color="#2563eb")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/loss-curve.png", dpi=150)
    print("Saved data/loss-curve.png")
    plt.close()


def plot_attention(model: TinyTransformer, word2idx: dict[str, int]):
    """Plot attention weights for a sample sentence."""
    sentence = "the cat sat on"
    tokens = [word2idx[w] for w in sentence.split()]
    x = torch.tensor(tokens).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        tok = model.token_emb(x)
        pos = model.pos_emb(torch.arange(x.shape[1]))
        h = tok + pos
        _, attn_weights = model.attention(
            h, h, h,
            attn_mask=model.causal_mask[:x.shape[1], :x.shape[1]],
            average_attn_weights=True,
        )

    weights = attn_weights.squeeze(0).numpy()
    words = sentence.split()

    plt.figure(figsize=(5, 4))
    plt.imshow(weights, cmap="Blues", vmin=0, vmax=1)
    plt.xticks(range(len(words)), words)
    plt.yticks(range(len(words)), words)
    plt.xlabel("Attends to")
    plt.ylabel("From")
    plt.title("Attention weights: 'the cat sat on'")
    plt.colorbar(label="Weight")
    plt.tight_layout()
    plt.savefig("data/attention.png", dpi=150)
    print("Saved data/attention.png")
    plt.close()


if __name__ == "__main__":
    with open("data/losses.json") as f:
        losses = json.load(f)
    with open("data/vocab.json") as f:
        word2idx = json.load(f)

    model = TinyTransformer(len(word2idx), embed_dim=64, context_length=4, num_heads=4)
    model.load_state_dict(torch.load("data/model.pt", weights_only=True))

    plot_loss(losses)
    plot_attention(model, word2idx)
