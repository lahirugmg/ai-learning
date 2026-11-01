"""
Step 7 — Visualize: plot the decision boundary, loss curve, and neuron activations.

Run:
    uv run python 07_visualize.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import importlib
_backprop_mod = importlib.import_module("04_backprop")
TrainableNetwork = _backprop_mod.TrainableNetwork


def load_model(path: str) -> TrainableNetwork:
    net = TrainableNetwork()
    weights = np.load(path)
    net.W1 = weights["W1"]
    net.b1 = weights["b1"]
    net.W2 = weights["W2"]
    net.b2 = weights["b2"]
    return net


def plot_loss_curve(history: list[dict]):
    """Plot training loss and accuracy over epochs."""
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    accs = [h["accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, losses, linewidth=2, color="#2563eb")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary cross-entropy loss")
    ax1.set_title("Training loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, accs, linewidth=2, color="#16a34a")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/loss-curve.png", dpi=150)
    print("Saved data/loss-curve.png")
    plt.close()


def plot_decision_boundary(net: TrainableNetwork):
    """Plot what the network learned: the decision boundary in 2D."""
    # Load training data
    data = np.genfromtxt("data/students.csv", delimiter=",", skip_header=2)
    X = data[:, :2]
    y = data[:, 2]

    # Create a grid of points covering the input space
    x1_range = np.linspace(0, 11, 200)
    x2_range = np.linspace(0, 11, 200)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()] / 10.0  # normalize like training

    # Predict on every grid point
    probs = net.forward(grid).reshape(xx1.shape)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color the background by prediction
    contour = ax.contourf(xx1, xx2, probs, levels=50, cmap="RdYlBu", alpha=0.7)
    ax.contour(xx1, xx2, probs, levels=[0.5], colors="black", linewidths=2)
    plt.colorbar(contour, ax=ax, label="P(pass)")

    # Plot the actual students
    fail = y == 0
    passed = y == 1
    ax.scatter(X[fail, 0], X[fail, 1], c="red", edgecolors="darkred",
               s=80, label="Failed", zorder=5)
    ax.scatter(X[passed, 0], X[passed, 1], c="blue", edgecolors="darkblue",
               s=80, label="Passed", zorder=5)

    ax.set_xlabel("Hours studied")
    ax.set_ylabel("Hours slept")
    ax.set_title("Decision boundary — black line is P(pass) = 0.5")
    ax.legend()
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    plt.tight_layout()
    plt.savefig("data/decision-boundary.png", dpi=150)
    print("Saved data/decision-boundary.png")
    plt.close()


def plot_hidden_neurons(net: TrainableNetwork):
    """Visualize what each hidden neuron 'sees'."""
    x1_range = np.linspace(0, 11, 200)
    x2_range = np.linspace(0, 11, 200)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()] / 10.0

    # Forward pass to get hidden activations
    net.forward(grid)
    activations = net.a1  # (40000, 4)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for i, ax in enumerate(axes):
        act = activations[:, i].reshape(xx1.shape)
        im = ax.imshow(act, extent=[0, 11, 0, 11], origin="lower",
                       cmap="viridis", aspect="auto")
        ax.set_title(f"Neuron {i+1}")
        ax.set_xlabel("Hours studied")
        if i == 0:
            ax.set_ylabel("Hours slept")
        plt.colorbar(im, ax=ax)

    plt.suptitle("What each hidden neuron responds to (ReLU activation)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig("data/hidden-neurons.png", dpi=150)
    print("Saved data/hidden-neurons.png")
    plt.close()


if __name__ == "__main__":
    net = load_model("data/model.npz")

    with open("data/history.json") as f:
        history = json.load(f)

    plot_loss_curve(history)
    plot_decision_boundary(net)
    plot_hidden_neurons(net)

