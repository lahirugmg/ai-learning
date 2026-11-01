"""
Step 5 — Train: repeat forward → loss → backward → update until the loss is low.

This is the full training loop. Every iteration:
  1. Forward pass:  compute predictions
  2. Compute loss:  how wrong are we?
  3. Backward pass:  compute gradients (which direction to adjust)
  4. Update weights:  nudge weights to reduce the loss

Run:
    uv run python 05_train.py
"""

import json

import numpy as np

import importlib
_backprop_mod = importlib.import_module("04_backprop")
TrainableNetwork = _backprop_mod.TrainableNetwork
sigmoid = _backprop_mod.sigmoid

_loss_mod = importlib.import_module("03_loss")
binary_cross_entropy = _loss_mod.binary_cross_entropy
accuracy = _loss_mod.accuracy


def train(net: TrainableNetwork, X: np.ndarray, y: np.ndarray,
          epochs: int = 1000, lr: float = 1.0) -> list[dict]:
    """Train the network and return metrics at each epoch."""
    history = []

    for epoch in range(epochs):
        # 1. Forward
        y_pred = net.forward(X)

        # 2. Loss
        loss = binary_cross_entropy(y, y_pred)
        acc = accuracy(y, y_pred)

        # 3. Backward
        net.backward(y)

        # 4. Update (gradient descent)
        net.W1 -= lr * net.dW1
        net.b1 -= lr * net.db1
        net.W2 -= lr * net.dW2
        net.b2 -= lr * net.db2

        history.append({"epoch": epoch + 1, "loss": loss, "accuracy": acc})

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  "
                  f"loss = {loss:.4f}  accuracy = {acc:.1%}")

    return history


if __name__ == "__main__":
    # Load and normalize data
    data = np.genfromtxt("data/students.csv", delimiter=",", skip_header=2)
    X = data[:, :2] / 10.0   # normalize to [0, 1]
    y = data[:, 2:3]

    net = TrainableNetwork()

    print(f"Training on {len(X)} students for 1000 epochs...\n")
    history = train(net, X, y, epochs=1000, lr=1.0)

    final = history[-1]
    print(f"\nFinal loss: {final['loss']:.4f}")
    print(f"Final accuracy: {final['accuracy']:.1%}")

    # Save the model weights
    np.savez("data/model.npz",
             W1=net.W1, b1=net.b1,
             W2=net.W2, b2=net.b2)

    # Save training history
    with open("data/history.json", "w") as f:
        json.dump(history, f)

    print("\nSaved model to data/model.npz")
    print("Saved training history to data/history.json")

    # Show some predictions
    print("\nSample predictions on training data:")
    y_pred = net.forward(X)
    print(f"  {'Studied':>7}  {'Slept':>5}  {'True':>4}  {'Predicted':>9}  {'Correct':>7}")
    print(f"  {'─'*7}  {'─'*5}  {'─'*4}  {'─'*9}  {'─'*7}")
    for i in range(len(X)):
        s, sl = X[i] * 10
        true = int(y[i, 0])
        pred = y_pred[i, 0]
        label = "✓" if (pred >= 0.5) == bool(true) else "✗"
        print(f"  {s:7.0f}  {sl:5.0f}  {true:4d}  {pred:9.4f}  {label:>7}")

