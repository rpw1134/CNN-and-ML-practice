"""
Test: MLP circle classification.

Generates a 2D dataset where class 1 = inside a circle, class 0 = outside.
This is not linearly separable, so it's a good sanity-check that the MLP
can learn a non-linear decision boundary.

Pass criterion: >= 90% accuracy on held-out test set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless — save to file instead of showing

from ml_practice_and_neural_net.ml_practice.neural_nets.networks.MLP import MLP
from ml_practice_and_neural_net.ml_practice.data_classes.NNLayerParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNHyperparameters import NNHyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNParameters import NNParameters
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_circle_data


# ── helpers ──────────────────────────────────────────────────────────────────


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def to_onehot(y: np.ndarray, n_classes: int = 2) -> np.ndarray:
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1.0
    return oh


def plot_decision_boundary(model: MLP, X: np.ndarray, y: np.ndarray,
                            save_path: Path) -> None:
    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    y_min, y_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    zz = np.argmax(probs, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, zz, alpha=0.3, cmap="RdBu")
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c="red",  label="outside", alpha=0.6)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c="blue", label="inside",  alpha=0.6)
    circle = plt.Circle((0, 0), 1.0, color="black", fill=False, linewidth=2, linestyle="--", label="true boundary")
    ax.add_patch(circle)
    ax.set_title("MLP — Circle Classification Decision Boundary")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Decision boundary plot saved → {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MLP Circle Classification Test")
    print("=" * 60)

    # ── data ──────────────────────────────────────────────────────
    print("\n1. Generating circle dataset...")
    X, y = generate_circle_data(n_samples=1000, radius=1.0, noise=0.05, random_state=42)
    Y_oh = to_onehot(y, n_classes=2)

    # manual 80/20 split (keep test set separate from MLP's internal val split)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    Y_train_oh   = to_onehot(y_train, n_classes=2)

    print(f"   Total samples : {len(X)}")
    print(f"   Train / test  : {len(X_train)} / {len(X_test)}")
    print(f"   Class balance : {np.mean(y == 1):.1%} inside, {np.mean(y == 0):.1%} outside")

    # ── architecture ──────────────────────────────────────────────
    print("\n2. Building MLP...")
    #   Input(2) → Hidden(16, relu) → Hidden(16, relu) → Output(2, sigmoid)
    #   No residuals on the first layer (dim mismatch 2→16 would add projection
    #   noise for such a small network; easier to disable for this test).
    layer_configs = [
        NNLayerParameters(width=32, input_dim=2,  activation="relu",    weight_init="he_normal",     sparsity=1.0, use_residuals=False),
        NNLayerParameters(width=32, input_dim=32, activation="relu",    weight_init="he_normal",     sparsity=1.0, use_residuals=True),
        NNLayerParameters(width=2,  input_dim=32, activation="sigmoid", weight_init="xavier_normal", sparsity=1.0, use_residuals=False),
    ]

    hyperparams = NNHyperparameters(
        learning_rate=0.01,
        epochs=300,
        batch_size=32,
        early_stopping_patience=30,
        regularization_type="l2",
        regularization_strength=0.0,
        num_iterations=0,
    )

    params = NNParameters(loss="cce")

    mlp = MLP(layer_configs=layer_configs, hyperparameters=hyperparams, params=params)
    print(f"   Layers        : {[f'{l.input_dim}→{l.output_dim}' for l in mlp.layers]}")
    print(f"   Loss          : {params.loss}")
    print(f"   Learning rate : {hyperparams.learning_rate}")
    print(f"   Epochs        : {hyperparams.epochs}  (patience={hyperparams.early_stopping_patience})")

    # ── training ──────────────────────────────────────────────────
    print("\n3. Training...")
    mlp.train(X_train, Y_train_oh)

    # ── evaluation ────────────────────────────────────────────────
    print("\n4. Evaluating on held-out test set...")
    test_probs = mlp.predict(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc   = accuracy(y_test, test_preds)

    train_probs = mlp.predict(X_train)
    train_preds = np.argmax(train_probs, axis=1)
    train_acc   = accuracy(y_train, train_preds)

    print(f"   Train accuracy : {train_acc:.2%}")
    print(f"   Test  accuracy : {test_acc:.2%}")

    # ── plot ──────────────────────────────────────────────────────
    print("\n5. Plotting decision boundary...")
    out_dir = Path(__file__).parent
    plot_decision_boundary(mlp, X_test, y_test, out_dir / "mlp_circle_boundary.png")

    # ── result ────────────────────────────────────────────────────
    PASS_THRESHOLD = 0.90
    print("\n" + "=" * 60)
    if test_acc >= PASS_THRESHOLD:
        print(f"✓ PASSED — test accuracy {test_acc:.2%} >= {PASS_THRESHOLD:.0%}")
    else:
        print(f"✗ FAILED — test accuracy {test_acc:.2%} < {PASS_THRESHOLD:.0%}")
    print("=" * 60)

