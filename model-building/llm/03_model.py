"""
Step 3 — Model: A tiny Transformer language model.

This is the same architecture as GPT, just much smaller. It has:
  • An embedding layer (words → vectors)
  • A positional encoding (so the model knows word order)
  • One Transformer block (self-attention + feed-forward)
  • A linear head (vectors → word probabilities)

All the maths (attention, softmax, layer norm) are explained in the README.

Run:
    uv run python 03_model.py
"""

import math

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    """A minimal GPT-style Transformer.

    Parameters:
        vocab_size:      how many unique words (determines embedding table size)
        embed_dim:       size of each word vector (d_model in the paper)
        context_length:  how many words the model sees at once
        num_heads:       how many attention heads (must divide embed_dim)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        context_length: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        # --- Embedding layers ---
        # Token embedding: each word id → a vector of `embed_dim` numbers
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        # Position embedding: each position (0,1,2,3) → a vector
        self.pos_emb = nn.Embedding(context_length, embed_dim)

        # --- Transformer block ---
        # Self-attention: lets each word look at the other words
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embed_dim)

        # Feed-forward: two linear layers with ReLU in between
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)

        # --- Output head ---
        # Projects the final vector back to vocab_size logits (one per word)
        self.head = nn.Linear(embed_dim, vocab_size)

        # Causal mask: prevents the model from peeking at future words
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, context_length) — token ids
        returns: (batch, vocab_size) — logits for the next word
        """
        B, T = x.shape

        # 1. Look up embeddings and add position information
        tok = self.token_emb(x)               # (B, T, embed_dim)
        pos = self.pos_emb(torch.arange(T, device=x.device))  # (T, embed_dim)
        h = tok + pos                         # (B, T, embed_dim)

        # 2. Self-attention (with causal mask so word i can only see words 0..i)
        attn_out, _ = self.attention(
            h, h, h, attn_mask=self.causal_mask[:T, :T]
        )
        h = self.ln1(h + attn_out)            # residual connection + layer norm

        # 3. Feed-forward
        ff_out = self.ff(h)
        h = self.ln2(h + ff_out)              # residual connection + layer norm

        # 4. Take the last position's vector and predict the next word
        last = h[:, -1, :]                    # (B, embed_dim)
        logits = self.head(last)              # (B, vocab_size)
        return logits


if __name__ == "__main__":
    import json

    with open("data/vocab.json") as f:
        word2idx = json.load(f)

    vocab_size = len(word2idx)
    model = TinyTransformer(vocab_size, embed_dim=64, context_length=4, num_heads=4)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model parameters: {total:,}")
    print()

    # Show the architecture
    print("Model architecture:")
    print(model)
    print()

    # Do one forward pass with random input
    dummy = torch.randint(0, vocab_size, (1, 4))
    logits = model(dummy)
    probs = torch.softmax(logits, dim=-1)
    print(f"Input shape:  {dummy.shape}  (1 batch × 4 words)")
    print(f"Output shape: {logits.shape}  (1 batch × {vocab_size} word probabilities)")
    print(f"Probabilities sum to: {probs.sum().item():.4f}")

