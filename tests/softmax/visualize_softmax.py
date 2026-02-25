"""
Visualize the SoftmaxRegressionModel decision boundaries on 3-class data.
Requires matplotlib.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_multiclass_linear_data
from ml_practice_and_neural_net.ml_practice.models.SoftmaxRegressionModel import SoftmaxRegressionModel
from ml_practice_and_neural_net.ml_practice.data_classes.ModelHyperparameters import ModelHyperparameters


def plot_decision_boundaries(model, X, y, title="Decision Boundaries"):
    """Plot the decision boundaries of a trained model."""
    # Create a mesh grid
    h = 0.1  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         edgecolors='black', s=50)
    plt.colorbar(scatter, label='Class')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    return plt


if __name__ == "__main__":
    print("Generating and visualizing SoftmaxRegressionModel...")

    # Generate dataset
    X, y = generate_multiclass_linear_data(
        n_samples=600,
        n_features=2,
        n_classes=3,
        separation=6.0,
        noise=0.4,
        random_state=42
    )

    # Train model
    hyperparams = ModelHyperparameters(learning_rate=0.1, epochs=2000, regularizer=None, training_method="gradient_descent")
    model = SoftmaxRegressionModel(hyperparams)
    model.fit(X, y)

    # Calculate accuracy
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred) * 100

    # Plot
    plot = plot_decision_boundaries(model, X, y,
                                   f"Softmax Regression Decision Boundaries\nAccuracy: {accuracy:.2f}%")

    print(f"Accuracy: {accuracy:.2f}%")
    print("Displaying plot...")

    try:
        plot.savefig('/Users/ryanwilliams/Projects/ml-practice-and-neural-net/softmax_decision_boundaries.png',
                    dpi=150, bbox_inches='tight')
        print("✅ Plot saved to: softmax_decision_boundaries.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    try:
        plot.show()
    except Exception as e:
        print(f"Could not display plot (no display available): {e}")
