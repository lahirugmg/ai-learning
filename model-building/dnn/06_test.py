"""
Step 6 — Test: evaluate the trained model on data it has never seen.

Training accuracy tells you how well the model memorized. Test accuracy
tells you how well it generalized — can it make correct predictions on
new students?

Run:
    uv run python 06_test.py
"""

import numpy as np

import importlib
_backprop_mod = importlib.import_module("04_backprop")
TrainableNetwork = _backprop_mod.TrainableNetwork

_loss_mod = importlib.import_module("03_loss")
binary_cross_entropy = _loss_mod.binary_cross_entropy
accuracy = _loss_mod.accuracy


def load_model(path: str) -> TrainableNetwork:
    """Load saved weights into a new network."""
    net = TrainableNetwork()
    weights = np.load(path)
    net.W1 = weights["W1"]
    net.b1 = weights["b1"]
    net.W2 = weights["W2"]
    net.b2 = weights["b2"]
    return net


if __name__ == "__main__":
    # Load the trained model
    net = load_model("data/model.npz")

    # Load test data (students the model has never seen)
    test_data = np.genfromtxt("data/students-test.csv", delimiter=",", skip_header=1)
    X_test = test_data[:, :2] / 10.0
    y_test = test_data[:, 2:3]

    # Also evaluate on training data for comparison
    train_data = np.genfromtxt("data/students.csv", delimiter=",", skip_header=2)
    X_train = train_data[:, :2] / 10.0
    y_train = train_data[:, 2:3]

    # Predictions
    y_pred_train = net.forward(X_train)
    y_pred_test = net.forward(X_test)

    train_loss = binary_cross_entropy(y_train, y_pred_train)
    test_loss = binary_cross_entropy(y_test, y_pred_test)
    train_acc = accuracy(y_train, y_pred_train)
    test_acc = accuracy(y_test, y_pred_test)

    print("=== Model evaluation ===\n")
    print(f"  {'':>10}  {'Loss':>8}  {'Accuracy':>8}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}")
    print(f"  {'Train':>10}  {train_loss:8.4f}  {train_acc:8.1%}")
    print(f"  {'Test':>10}  {test_loss:8.4f}  {test_acc:8.1%}")
    print()

    if test_acc >= train_acc - 0.1:
        print("  The model generalizes well — test accuracy is close to train.")
    elif test_acc < train_acc - 0.2:
        print("  ⚠ Possible overfitting — test accuracy is much lower than train.")
    print()

    # Detailed predictions on test set
    print("Test set predictions:\n")
    print(f"  {'Studied':>7}  {'Slept':>5}  {'True':>4}  {'Predicted':>9}  {'Result':>7}")
    print(f"  {'─'*7}  {'─'*5}  {'─'*4}  {'─'*9}  {'─'*7}")
    for i in range(len(X_test)):
        s, sl = X_test[i] * 10
        true = int(y_test[i, 0])
        pred = y_pred_test[i, 0]
        label = "✓" if (pred >= 0.5) == bool(true) else "✗"
        print(f"  {s:7.0f}  {sl:5.0f}  {true:4d}  {pred:9.4f}  {label:>7}")

    # Confusion matrix
    pred_labels = (y_pred_test >= 0.5).astype(int).flatten()
    true_labels = y_test.flatten().astype(int)

    tp = np.sum((pred_labels == 1) & (true_labels == 1))
    tn = np.sum((pred_labels == 0) & (true_labels == 0))
    fp = np.sum((pred_labels == 1) & (true_labels == 0))
    fn = np.sum((pred_labels == 0) & (true_labels == 1))

    print(f"\nConfusion matrix:")
    print(f"                  Predicted")
    print(f"                  Fail  Pass")
    print(f"  Actual  Fail  [ {tn:2d}    {fp:2d} ]")
    print(f"          Pass  [ {fn:2d}    {tp:2d} ]")
    print(f"\n  True positives: {tp}  |  True negatives: {tn}")
    print(f"  False positives: {fp}  |  False negatives: {fn}")

