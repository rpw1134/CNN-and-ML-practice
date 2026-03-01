"""
Test: MLP sine regression.

Generates noisy samples of y = sin(x) over [-π, π] and trains an MLP to
regress the underlying curve. This verifies the network can learn a smooth
non-linear function using MSE loss.

Pass criterion: R² >= 0.95 on held-out test set.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from ml_practice_and_neural_net.ml_practice.neural_nets.networks.MLP import MLP
from ml_practice_and_neural_net.ml_practice.data_classes.NNLayerParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNHyperparameters import NNHyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNParameters import NNParameters
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_sine_data


# ── helpers ──────────────────────────────────────────────────────────────────


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)


def plot_fit(model: MLP, X_test: np.ndarray, y_test: np.ndarray,
             save_path: Path) -> None:
    x_line = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
    y_true = np.sin(x_line)
    y_pred = model.predict(x_line)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(X_test, y_test, s=10, alpha=0.4, color="steelblue", label="test samples")
    ax.plot(x_line, y_true, color="black",  linewidth=2,   label="true sin(x)")
    ax.plot(x_line, y_pred, color="tomato", linewidth=2, linestyle="--", label="MLP prediction")
    ax.set_title("MLP — Sine Regression")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"   Fit plot saved → {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MLP Sine Regression Test")
    print("=" * 60)

    # ── data ──────────────────────────────────────────────────────
    print("\n1. Generating sine dataset...")
    X, Y = generate_sine_data(n_samples=800, noise=0.05, random_state=42)

    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    Y_train, Y_test = Y[idx[:split]], Y[idx[split:]]

    print(f"   Total samples : {len(X)}")
    print(f"   Train / test  : {len(X_train)} / {len(X_test)}")
    print(f"   x range       : [{X.min():.3f}, {X.max():.3f}]")
    print(f"   y range       : [{Y.min():.3f}, {Y.max():.3f}]")

    # ── architecture ──────────────────────────────────────────────
    # Regression: output layer is width=1 with identity activation + MSE loss.
    # tanh hidden units suit a sine target well (both bounded, smooth, odd).
    print("\n2. Building MLP...")
    layer_configs = [
        NNLayerParameters(width=32, input_dim=1,  activation="hyperbolic_tan", weight_init="xavier_normal", sparsity=1.0, use_residuals=False),
        NNLayerParameters(width=32, input_dim=32, activation="hyperbolic_tan", weight_init="xavier_normal", sparsity=1.0, use_residuals=True),
        NNLayerParameters(width=1,  input_dim=32, activation="identity",       weight_init="xavier_normal", sparsity=1.0, use_residuals=False),
    ]

    hyperparams = NNHyperparameters(
        learning_rate=0.005,
        epochs=500,
        batch_size=32,
        early_stopping_patience=30,
        regularization_type="l2",
        regularization_strength=0.0,
        num_iterations=0,
    )

    params = NNParameters(loss="mse")

    mlp = MLP(layer_configs=layer_configs, hyperparameters=hyperparams, params=params)
    print(f"   Layers        : {[f'{l.input_dim}→{l.output_dim}' for l in mlp.layers]}")
    print(f"   Loss          : {params.loss}")
    print(f"   Learning rate : {hyperparams.learning_rate}")
    print(f"   Epochs        : {hyperparams.epochs}  (patience={hyperparams.early_stopping_patience})")

    # ── training ──────────────────────────────────────────────────
    print("\n3. Training...")
    mlp.train(X_train, Y_train)

    # ── evaluation ────────────────────────────────────────────────
    print("\n4. Evaluating on held-out test set...")
    train_preds = mlp.predict(X_train)
    test_preds  = mlp.predict(X_test)

    train_r2  = r2_score(Y_train, train_preds)
    test_r2   = r2_score(Y_test,  test_preds)
    test_rmse = float(np.sqrt(np.mean((Y_test - test_preds) ** 2)))

    print(f"   Train R²   : {train_r2:.4f}")
    print(f"   Test  R²   : {test_r2:.4f}")
    print(f"   Test  RMSE : {test_rmse:.4f}")

    # ── plot ──────────────────────────────────────────────────────
    print("\n5. Plotting fit...")
    out_dir = Path(__file__).parent
    plot_fit(mlp, X_test, Y_test, out_dir / "mlp_sine_fit.png")

    # ── result ────────────────────────────────────────────────────
    PASS_THRESHOLD = 0.95
    print("\n" + "=" * 60)
    if test_r2 >= PASS_THRESHOLD:
        print(f"✓ PASSED — test R² {test_r2:.4f} >= {PASS_THRESHOLD}")
    else:
        print(f"✗ FAILED — test R² {test_r2:.4f} < {PASS_THRESHOLD}")
    print("=" * 60)

