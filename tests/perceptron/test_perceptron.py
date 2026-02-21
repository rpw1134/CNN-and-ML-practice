"""
Test script for Perceptron model on synthetic linearly separable data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_simple_linear_data
from ml_practice_and_neural_net.ml_practice.models.Perceptron import Perceptron
from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term


def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


if __name__ == "__main__":
    print("="*70)
    print("Testing Perceptron Model on Linearly Separable Data")
    print("="*70)

    # Generate linearly separable dataset
    print("\n1. Generating linearly separable dataset...")
    X, y = generate_simple_linear_data(
        n_samples=200,
        n_features=2,
        separation=2.0,
        noise=0.5,
        random_state=42
    )

    # Convert labels from {0, 1} to {-1, +1} for perceptron
    y = 2 * y - 1

    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Class distribution:")
    print(f"     Class -1: {np.sum(y == -1)} samples")
    print(f"     Class +1: {np.sum(y == 1)} samples")

    # Split into train and test sets
    train_size = int(0.8 * len(X))
    indices = np.random.RandomState(42).permutation(len(X))
    train_idx, test_idx = indices[:train_size], indices[train_size:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"\n2. Train/test split:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")

    # Train the perceptron
    print("\n3. Training Perceptron...")
    hyperparameters = Hyperparameters(
        learning_rate=0.1,
        epochs=100,
        regularizer=None,
        lambda_reg=0.0,
        training_method=None
    )

    perceptron = Perceptron(hyperparameters)
    perceptron.fit(X_train, y_train)

    print(f"   Training complete!")
    print(f"   Learned weights shape: {perceptron.weights.shape}")
    print(f"   Learned weights: {perceptron.weights.flatten()}")

    # Make predictions on training set
    print("\n4. Evaluating on training set...")
    X_train_bias = add_data_bias_term(X_train)
    y_train_pred = perceptron.predict(X_train_bias)
    train_accuracy = calculate_accuracy(y_train, y_train_pred)
    print(f"   Training Accuracy: {train_accuracy * 100:.2f}%")

    # Make predictions on test set
    print("\n5. Evaluating on test set...")
    X_test_bias = add_data_bias_term(X_test)
    y_test_pred = perceptron.predict(X_test_bias)
    test_accuracy = calculate_accuracy(y_test, y_test_pred)
    print(f"   Test Accuracy: {test_accuracy * 100:.2f}%")

    # Visualize decision boundary (for 2D data)
    if X.shape[1] == 2:
        print("\n6. Creating visualization...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot training data
        ax1 = axes[0]
        ax1.scatter(X_train[y_train.flatten() == -1, 0],
                   X_train[y_train.flatten() == -1, 1],
                   c='blue', marker='o', label='Class -1', alpha=0.6, edgecolors='k')
        ax1.scatter(X_train[y_train.flatten() == 1, 0],
                   X_train[y_train.flatten() == 1, 1],
                   c='red', marker='s', label='Class +1', alpha=0.6, edgecolors='k')

        # Plot decision boundary
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

        # Calculate decision boundary line: w0 + w1*x1 + w2*x2 = 0
        # Solving for x2: x2 = -(w0 + w1*x1) / w2
        w0, w1, w2 = perceptron.weights.flatten()
        has_boundary = abs(w2) > 1e-6
        if has_boundary:
            x_line = np.array([x_min, x_max])
            y_line = -(w0 + w1 * x_line) / w2
            ax1.plot(x_line, y_line, 'g-', linewidth=2, label='Decision Boundary')

        ax1.set_xlabel('Feature 1', fontsize=12)
        ax1.set_ylabel('Feature 2', fontsize=12)
        ax1.set_title(f'Training Data\n(Accuracy: {train_accuracy*100:.2f}%)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot test data
        ax2 = axes[1]
        ax2.scatter(X_test[y_test.flatten() == -1, 0],
                   X_test[y_test.flatten() == -1, 1],
                   c='blue', marker='o', label='Class -1', alpha=0.6, edgecolors='k')
        ax2.scatter(X_test[y_test.flatten() == 1, 0],
                   X_test[y_test.flatten() == 1, 1],
                   c='red', marker='s', label='Class +1', alpha=0.6, edgecolors='k')

        # Plot decision boundary (same as training)
        if has_boundary:
            x_line = np.array([x_min, x_max])
            y_line = -(w0 + w1 * x_line) / w2
            ax2.plot(x_line, y_line, 'g-', linewidth=2, label='Decision Boundary')

        # Mark misclassified points
        misclassified_test = y_test_pred.flatten() != y_test.flatten()
        if np.any(misclassified_test):
            ax2.scatter(X_test[misclassified_test, 0],
                       X_test[misclassified_test, 1],
                       c='yellow', marker='x', s=200, linewidths=3,
                       label='Misclassified', zorder=5)

        ax2.set_xlabel('Feature 1', fontsize=12)
        ax2.set_ylabel('Feature 2', fontsize=12)
        ax2.set_title(f'Test Data\n(Accuracy: {test_accuracy*100:.2f}%)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save the figure
        output_path = Path(__file__).parent / "perceptron_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   Visualization saved to: {output_path}")

        plt.show()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test Accuracy:     {test_accuracy * 100:.2f}%")
    print(f"Learned weights:   {perceptron.weights.flatten()}")
    print("="*70)
    print("\nPerceptron test complete!")

