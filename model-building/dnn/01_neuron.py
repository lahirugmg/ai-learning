"""
Step 1 — A single neuron: the building block of every neural network.

A neuron does three things:
  1. Multiply each input by a weight       (how important is this input?)
  2. Add them up, plus a bias              (the weighted sum)
  3. Apply an activation function          (introduce nonlinearity)

This is the same thing a logistic regression does — a neural network is just
many of these stacked together.

Run:
    uv run python 01_neuron.py
"""

import numpy as np


# ── Activation functions ──────────────────────────────────────────────
# These squash the weighted sum into a useful range.

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash any number into the range (0, 1).

    Big positive → close to 1
    Big negative → close to 0
    Zero         → exactly 0.5

    Formula: σ(z) = 1 / (1 + e^(-z))
    """
    return 1 / (1 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    """Keep positives, zero out negatives.

    Positive → unchanged
    Negative → 0

    Formula: ReLU(z) = max(0, z)
    """
    return np.maximum(0, z)


# ── A single neuron ──────────────────────────────────────────────────

def neuron(inputs: np.ndarray, weights: np.ndarray, bias: float,
           activation=sigmoid) -> float:
    """Compute the output of one neuron.

    Steps:
      1. weighted_sum = (input₁ × weight₁) + (input₂ × weight₂) + ... + bias
      2. output = activation(weighted_sum)

    Args:
        inputs:     array of input values, e.g. [hours_studied, hours_slept]
        weights:    one weight per input, learned during training
        bias:       a single number, also learned during training
        activation: the function applied after the weighted sum
    """
    weighted_sum = np.dot(inputs, weights) + bias
    return activation(weighted_sum)


if __name__ == "__main__":
    # A student who studied 7 hours and slept 6 hours
    inputs = np.array([7.0, 6.0])

    # Made-up weights (training would find good ones)
    weights = np.array([0.5, 0.3])
    bias = -3.0

    # Step by step
    weighted_sum = np.dot(inputs, weights) + bias
    output = sigmoid(weighted_sum)

    print("=== A single neuron ===\n")
    print(f"Inputs:  hours_studied={inputs[0]:.0f}, hours_slept={inputs[1]:.0f}")
    print(f"Weights: w₁={weights[0]}, w₂={weights[1]}")
    print(f"Bias:    b={bias}")
    print()
    print(f"Step 1 — weighted sum:")
    print(f"  ({inputs[0]} × {weights[0]}) + ({inputs[1]} × {weights[1]}) + {bias}")
    print(f"  = {inputs[0]*weights[0]} + {inputs[1]*weights[1]} + {bias}")
    print(f"  = {weighted_sum}")
    print()
    print(f"Step 2 — apply sigmoid:")
    print(f"  σ({weighted_sum}) = 1 / (1 + e^(-{weighted_sum}))")
    print(f"  = {output:.4f}")
    print()
    print(f"Interpretation: {output:.1%} chance of passing")
    print()

    # Show how different activations behave
    print("=== Activation functions on values from -3 to +3 ===\n")
    values = np.array([-3, -1, 0, 1, 3], dtype=float)
    print(f"  {'Input':>6}  {'Sigmoid':>8}  {'ReLU':>6}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}")
    for v in values:
        print(f"  {v:6.1f}  {sigmoid(v):8.4f}  {relu(v):6.1f}")

