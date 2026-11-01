"""
Step 3 — Loss function: measuring how wrong the predictions are.

The loss function is a single number that says "how bad is the model right now?"
Training means making this number smaller.

For binary classification (pass/fail), we use Binary Cross-Entropy (BCE):

    loss = -1/N × Σ [ y·log(ŷ) + (1-y)·log(1-ŷ) ]

where y is the true label (0 or 1) and ŷ is the predicted probability.

Why this formula?
  • If y=1 and ŷ=0.9: loss = -log(0.9) = 0.105   (low, good prediction)
  • If y=1 and ŷ=0.1: loss = -log(0.1) = 2.303   (high, bad prediction)
  • If y=0 and ŷ=0.1: loss = -log(0.9) = 0.105   (low, good prediction)
  • If y=0 and ŷ=0.9: loss = -log(0.1) = 2.303   (high, bad prediction)

Confident AND correct → low loss. Confident AND wrong → very high loss.

Run:
    uv run python 03_loss.py
"""

import numpy as np

import importlib
_net_mod = importlib.import_module("02_network")
Network = _net_mod.Network


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the average binary cross-entropy loss.

    Args:
        y_true: true labels, shape (N, 1), each value is 0 or 1
        y_pred: predicted probabilities, shape (N, 1), each in (0, 1)

    Returns:
        A single number — lower is better.
    """
    # Clip predictions to avoid log(0) which is -infinity
    eps = 1e-7
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # The formula
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions (threshold at 0.5)."""
    predicted_labels = (y_pred >= 0.5).astype(int)
    return np.mean(predicted_labels == y_true)


if __name__ == "__main__":
    # Show what the loss looks like for different predictions
    print("=== Binary cross-entropy loss ===\n")
    print("When the true label is 1 (should pass):")
    print(f"  {'Prediction':>10}  {'Loss':>8}  Interpretation")
    print(f"  {'─'*10}  {'─'*8}  {'─'*20}")
    for p in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]:
        loss = -np.log(p)
        bar = "█" * int(loss * 5)
        print(f"  {p:10.2f}  {loss:8.3f}  {bar}")

    print()
    print("When the true label is 0 (should fail):")
    print(f"  {'Prediction':>10}  {'Loss':>8}  Interpretation")
    print(f"  {'─'*10}  {'─'*8}  {'─'*20}")
    for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        loss = -np.log(1 - p)
        bar = "█" * int(loss * 5)
        print(f"  {p:10.2f}  {loss:8.3f}  {bar}")

    # Load data and compute loss for the untrained network
    print()
    print("=== Loss on the student dataset (untrained network) ===\n")

    data = np.genfromtxt("data/students.csv", delimiter=",", skip_header=2)
    X = data[:, :2] / 10.0  # normalize to [0, 1]
    y = data[:, 2:3]        # passed (0 or 1)

    net = Network()
    y_pred = net.forward(X)

    loss = binary_cross_entropy(y, y_pred)
    acc = accuracy(y, y_pred)

    print(f"  Predictions range: {y_pred.min():.4f} to {y_pred.max():.4f}")
    print(f"  Loss (BCE):        {loss:.4f}")
    print(f"  Accuracy:          {acc:.1%}")
    print(f"  (Random guessing would give ~50% accuracy and ~0.69 loss)")
