"""
Test script demonstrating Perceptron limitations on non-linearly separable data (XOR).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_xor_data
from ml_practice_and_neural_net.ml_practice.models.Perceptron import Perceptron
from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term


def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    print("="*70)
    print("Testing Perceptron on XOR Data (Non-Linearly Separable)")
    print("="*70)
    print("\nNote: This test demonstrates the perceptron's limitation.")
    print("The XOR problem is NOT linearly separable, so the perceptron")
    print("cannot achieve perfect classification.\n")

    # Generate XOR dataset
    print("1. Generating XOR dataset (non-linearly separable)...")
    X, y = generate_xor_data(
        n_samples=200,
        noise=0.3,
        random_state=42
    )

    # Convert labels from {0, 1} to {-1, +1} for perceptron
    y = 2 * y - 1

    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution:")
    print(f"     Class -1: {np.sum(y == -1)} samples")
    print(f"     Class +1: {np.sum(y == 1)} samples")

    # Train the perceptron
    print("\n2. Training Perceptron...")
    hyperparameters = Hyperparameters(
        learning_rate=0.01,
        epochs=1000,
        regularizer=None,
        lambda_reg=0.0,
        training_method=None
    )

    perceptron = Perceptron(hyperparameters)
    perceptron.fit(X, y)

    print(f"   Training complete!")
    print(f"   Learned weights: {perceptron.weights.flatten()}")

    # Make predictions
    print("\n3. Evaluating on training set...")
    X_bias = add_data_bias_term(X)
    y_pred = perceptron.predict(X_bias)
    accuracy = calculate_accuracy(y, y_pred)
    print(f"   Training Accuracy: {accuracy * 100:.2f}%")
    print(f"   (Expected to be around 50% - random chance - due to XOR)")

    # Visualize
    print("\n4. Creating visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: True XOR pattern
    ax1 = axes[0]
    ax1.scatter(X[y.flatten() == -1, 0],
               X[y.flatten() == -1, 1],
               c='blue', marker='o', label='Class -1', alpha=0.6, edgecolors='k', s=50)
    ax1.scatter(X[y.flatten() == 1, 0],
               X[y.flatten() == 1, 1],
               c='red', marker='s', label='Class +1', alpha=0.6, edgecolors='k', s=50)
    ax1.set_xlabel('Feature 1', fontsize=12)
    ax1.set_ylabel('Feature 2', fontsize=12)
    ax1.set_title('True XOR Pattern\n(Non-Linearly Separable)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Perceptron's attempt with decision boundary
    ax2 = axes[1]

    # Create a mesh for decision boundary visualization
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_bias = add_data_bias_term(mesh_points)
    Z = perceptron.predict(mesh_points_bias)
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax2.contourf(xx, yy, Z, alpha=0.3, levels=[-1, 0, 1], colors=['blue', 'red'])

    # Plot data points
    ax2.scatter(X[y.flatten() == -1, 0],
               X[y.flatten() == -1, 1],
               c='blue', marker='o', label='Class -1', alpha=0.7, edgecolors='k', s=50)
    ax2.scatter(X[y.flatten() == 1, 0],
               X[y.flatten() == 1, 1],
               c='red', marker='s', label='Class +1', alpha=0.7, edgecolors='k', s=50)

    # Mark misclassified points
    misclassified = y_pred.flatten() != y.flatten()
    if np.any(misclassified):
        ax2.scatter(X[misclassified, 0],
                   X[misclassified, 1],
                   c='yellow', marker='x', s=200, linewidths=3,
                   label=f'Misclassified ({np.sum(misclassified)})', zorder=5)

    # Plot decision boundary line
    w0, w1, w2 = perceptron.weights.flatten()
    if abs(w2) > 1e-6:
        x_line = np.array([x_min, x_max])
        y_line = -(w0 + w1 * x_line) / w2
        ax2.plot(x_line, y_line, 'g-', linewidth=3, label='Decision Boundary')

    ax2.set_xlabel('Feature 1', fontsize=12)
    ax2.set_ylabel('Feature 2', fontsize=12)
    ax2.set_title(f'Perceptron Classification\n(Accuracy: {accuracy*100:.2f}%)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # Save the figure
    output_path = Path(__file__).parent / "perceptron_xor_limitation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Visualization saved to: {output_path}")

    plt.show()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Misclassified: {np.sum(misclassified)} / {len(y)} samples")
    print("\nConclusion: The perceptron cannot solve the XOR problem because")
    print("it is NOT linearly separable. A linear decision boundary cannot")
    print("separate the two classes in XOR data.")
    print("="*70)

