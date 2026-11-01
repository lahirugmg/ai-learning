"""
Step 4 — Backpropagation: teaching the network by computing gradients.

Backpropagation answers the question: "If I nudge this weight a tiny bit,
how much does the loss change?" That rate of change is the gradient.

The chain rule from calculus makes this tractable. Instead of computing the
effect of each weight on the final loss directly, we work backwards:

    Loss ← Output layer ← Hidden layer ← Input

At each layer, we compute:
    dL/dW = (dL/dz) × (dz/dW)

This script adds backward() to the Network, which computes all gradients
in one pass.

Run:
    uv run python 04_backprop.py
"""

import numpy as np

import importlib
_net_mod = importlib.import_module("02_network")
sigmoid = _net_mod.sigmoid
relu = _net_mod.relu


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid: σ'(z) = σ(z) × (1 - σ(z))

    If the output is close to 0 or 1 (confident), the gradient is small.
    If the output is near 0.5 (uncertain), the gradient is large.
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0, else 0.

    Gradient flows through for positive inputs, stops for negative inputs.
    """
    return (z > 0).astype(float)


class TrainableNetwork:
    """Network with forward AND backward passes."""

    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 1) * np.sqrt(2 / 4)
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass — same as before, but saves intermediates."""
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, y_true: np.ndarray) -> dict:
        """Backward pass — compute gradients for all weights and biases.

        Uses the chain rule to propagate the error from the output back
        through each layer.

        Args:
            y_true: true labels, shape (N, 1)

        Returns:
            dict of gradients: dW1, db1, dW2, db2
        """
        N = y_true.shape[0]
        eps = 1e-7
        y_pred = np.clip(self.a2, eps, 1 - eps)

        # ── Output layer gradients ──
        # dL/da2 = derivative of BCE with respect to predictions
        dL_da2 = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / N

        # dL/dz2 = dL/da2 × da2/dz2 (sigmoid derivative)
        dz2 = dL_da2 * sigmoid_derivative(self.z2)  # (N, 1)

        # dL/dW2 = a1ᵀ × dz2
        self.dW2 = self.a1.T @ dz2                  # (4, 1)
        self.db2 = np.sum(dz2, axis=0)               # (1,)

        # ── Hidden layer gradients ──
        # dL/da1 = dz2 × W2ᵀ  (propagate error backwards)
        dL_da1 = dz2 @ self.W2.T                     # (N, 4)

        # dL/dz1 = dL/da1 × da1/dz1 (ReLU derivative)
        dz1 = dL_da1 * relu_derivative(self.z1)      # (N, 4)

        # dL/dW1 = Xᵀ × dz1
        self.dW1 = self.X.T @ dz1                    # (2, 4)
        self.db1 = np.sum(dz1, axis=0)                # (4,)

        return {"dW1": self.dW1, "db1": self.db1,
                "dW2": self.dW2, "db2": self.db2}


if __name__ == "__main__":
    # Load data
    data = np.genfromtxt("data/students.csv", delimiter=",", skip_header=2)
    X = data[:, :2]
    y = data[:, 2:3]

    # Normalize inputs to [0, 1] so gradients are well-scaled
    X = X / 10.0

    net = TrainableNetwork()

    # Forward pass
    y_pred = net.forward(X)

    # Backward pass
    grads = net.backward(y)

    print("=== Backpropagation ===\n")
    print("Gradients tell us how to adjust each weight to reduce the loss.\n")

    print("Gradient shapes (same shape as the weights):")
    for name, grad in grads.items():
        print(f"  {name}: {grad.shape}")
    print()

    print("Gradient values for W2 (hidden→output):")
    print(f"  dW2 = {grads['dW2'].flatten()}")
    print(f"  Meaning: nudge each weight in the opposite direction of its gradient")
    print()

    # Show what one gradient descent step looks like
    lr = 0.1
    print(f"One gradient descent step (learning rate = {lr}):")
    print(f"  W2 before: {net.W2.flatten()}")
    W2_new = net.W2 - lr * net.dW2
    print(f"  W2 after:  {W2_new.flatten()}")
    print()

    # Show the loss before and after one step
    from importlib import import_module
    _loss_mod = import_module("03_loss")
    bce = _loss_mod.binary_cross_entropy

    loss_before = bce(y, y_pred)

    # Apply the update
    net.W1 -= lr * net.dW1
    net.b1 -= lr * net.db1
    net.W2 -= lr * net.dW2
    net.b2 -= lr * net.db2

    y_pred_after = net.forward(X)
    loss_after = bce(y, y_pred_after)

    print(f"Loss before step: {loss_before:.4f}")
    print(f"Loss after step:  {loss_after:.4f}")
    print(f"Improvement:      {loss_before - loss_after:.4f}")
    print(f"\nThe loss went {'down ✓' if loss_after < loss_before else 'up ✗'}! "
          f"Repeat this 1000 times and the network learns.")

