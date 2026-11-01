"""
Step 4 — Train: Teach the model to predict the next word.

Training loop:
  1. Pick a batch of (context, target) pairs from the dataset.
  2. Feed the context through the model → get predicted probabilities.
  3. Measure how wrong the prediction is (cross-entropy loss).
  4. Compute gradients (backprop) and update the weights (optimizer step).
  5. Repeat until the loss is low.

Run:
    uv run python 04_train.py
"""

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import importlib

_dataset_mod = importlib.import_module("02_dataset")
NextWordDataset = _dataset_mod.NextWordDataset

_model_mod = importlib.import_module("03_model")
TinyTransformer = _model_mod.TinyTransformer


def train(
    model: TinyTransformer,
    dataset: NextWordDataset,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.003,
) -> list[float]:
    """Train the model and return the loss at each epoch."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            logits = model(x)           # forward pass
            loss = loss_fn(logits, y)   # how wrong is the prediction?
            optimizer.zero_grad()       # reset gradients
            loss.backward()             # compute gradients (backprop)
            optimizer.step()            # update weights
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss = {avg_loss:.4f}")

    return losses


if __name__ == "__main__":
    with open("data/tokens.json") as f:
        tokens = json.load(f)
    with open("data/vocab.json") as f:
        word2idx = json.load(f)

    dataset = NextWordDataset(tokens, context_length=4)
    model = TinyTransformer(len(word2idx), embed_dim=64, context_length=4, num_heads=4)

    print(f"Training on {len(dataset)} examples for 50 epochs...\n")
    losses = train(model, dataset)

    # Save the trained model
    torch.save(model.state_dict(), "data/model.pt")
    with open("data/losses.json", "w") as f:
        json.dump(losses, f)
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print("Saved model to data/model.pt")
