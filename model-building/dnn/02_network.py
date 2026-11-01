"""
Step 2 — Build a network: stack neurons into layers.

A single neuron draws a straight line through the data. To learn curved
boundaries, we stack neurons into layers:

  Input layer  →  Hidden layer  →  Output layer
  (2 numbers)     (4 neurons)      (1 neuron)

Each neuron in the hidden layer learns a different straight line. The output
neuron combines them into a curved decision boundary.

All the weights and biases are stored in matrices so we can compute a
whole layer at once using matrix multiplication.

Run:
    uv run python 02_network.py
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


class Network:
    """A simple feedforward neural network.

    Architecture: Input(2) → Hidden(4, ReLU) → Output(1, Sigmoid)

    Attributes:
        W1: weights from input to hidden   — shape (2, 4)
        b1: biases for hidden layer        — shape (4,)
        W2: weights from hidden to output  — shape (4, 1)
        b2: bias for output layer          — shape (1,)
    """

    def __init__(self):
        # Initialize weights randomly (small values).
        # He initialization: scale by sqrt(2/fan_in) for ReLU layers.
        np.random.seed(42)
        self.W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)  # 2 inputs → 4 hidden
        self.b1 = np.zeros(4)
        self.W2 = np.random.randn(4, 1) * np.sqrt(2 / 4)  # 4 hidden → 1 output
        self.b2 = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: compute prediction for each input row.

        X: shape (N, 2) — N students, 2 features each

        Returns: shape (N, 1) — predicted probability for each student
        """
        # Hidden layer: z1 = X·W1 + b1, then ReLU
        self.z1 = X @ self.W1 + self.b1        # (N, 4)
        self.a1 = relu(self.z1)                 # (N, 4)

        # Output layer: z2 = a1·W2 + b2, then sigmoid
        self.z2 = self.a1 @ self.W2 + self.b2   # (N, 1)
        self.a2 = sigmoid(self.z2)               # (N, 1)

        return self.a2

    def count_params(self) -> int:
        """Total number of trainable parameters."""
        return (self.W1.size + self.b1.size +
                self.W2.size + self.b2.size)


if __name__ == "__main__":
    net = Network()

    print("=== Network architecture ===\n")
    print("  Input layer:   2 features (hours_studied, hours_slept)")
    print(f"  Hidden layer:  4 neurons, ReLU activation")
    print(f"  Output layer:  1 neuron, sigmoid activation")
    print(f"  Total parameters: {net.count_params()}")
    print()

    # Show the shapes
    print("Weight shapes:")
    print(f"  W1: {net.W1.shape}  (2 inputs × 4 hidden neurons)")
    print(f"  b1: {net.b1.shape}  (one bias per hidden neuron)")
    print(f"  W2: {net.W2.shape}  (4 hidden neurons × 1 output)")
    print(f"  b2: {net.b2.shape}  (one bias for output)")
    print()

    # Forward pass with one student
    student = np.array([[7.0, 6.0]])
    prediction = net.forward(student)
    print(f"Student: studied={student[0,0]:.0f}h, slept={student[0,1]:.0f}h")
    print(f"Prediction (untrained): {prediction[0,0]:.4f}")
    print(f"(This is basically random — training will fix it)")
    print()

    # Forward pass with multiple students
    students = np.array([
        [1.0, 10.0],  # little study, lots of sleep
        [5.0, 7.0],   # moderate both
        [9.0, 5.0],   # lots of study, less sleep
    ])
    preds = net.forward(students)
    print("Batch prediction (3 students at once):")
    for i, (s, p) in enumerate(zip(students, preds)):
        print(f"  Student {i+1}: studied={s[0]:.0f}h, slept={s[1]:.0f}h "
              f"→ P(pass)={p[0]:.4f}")

