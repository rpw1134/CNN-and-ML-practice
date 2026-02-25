"""
Visualize LinearRegressionModel on simple 1D and 2D datasets.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import (
    generate_linear_regression_data,
    generate_polynomial_regression_data
)
from ml_practice_and_neural_net.ml_practice.models.LinearRegressionModel import LinearRegressionModel
from ml_practice_and_neural_net.ml_practice.data_classes.ModelHyperparameters import ModelHyperparameters


def plot_1d_regression():
    """Test and visualize linear regression on 1D data."""
    print("\n" + "="*70)
    print("1D Linear Regression Visualization")
    print("="*70)

    # Generate 1D data
    X, y, true_params = generate_linear_regression_data(
        n_samples=100,
        n_features=1,
        noise=2.0,
        random_state=42
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} feature")
    print(f"True parameters: intercept={true_params[0,0]:.3f}, slope={true_params[1,0]:.3f}")

    # Train model
    hyperparams = ModelHyperparameters(learning_rate=0.1, num_training_iterations=500)
    model = LinearRegressionModel(hyperparams)
    model.fit(X, y)

    print(f"Learned parameters: intercept={model.weights[0,0]:.3f}, slope={model.weights[1,0]:.3f}")
    print(f"R² score: {model.calculate_r2_score(X, y):.4f}")

    # Create predictions for plotting
    X_plot = np.linspace(X.min() - 1, X.max() + 1, 200).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    # True line
    y_true = X_plot * true_params[1, 0] + true_params[0, 0]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Fit
    ax1.scatter(X, y, alpha=0.6, s=30, label='Training data')
    ax1.plot(X_plot, y_true, 'g--', lw=2, label='True relationship')
    ax1.plot(X_plot, y_plot, 'r-', lw=2, label='Learned fit')
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Linear Regression Fit (1D)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Residuals
    y_pred = model.predict(X)
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=30)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/linear_regression/linear_regression_1d.png', dpi=150, bbox_inches='tight')
    print("Saved: tests/linear_regression/linear_regression_1d.png")


def plot_2d_regression():
    """Test and visualize linear regression on 2D data."""
    print("\n" + "="*70)
    print("2D Linear Regression Visualization")
    print("="*70)

    # Generate 2D data
    X, y, true_params = generate_linear_regression_data(
        n_samples=200,
        n_features=2,
        noise=1.0,
        random_state=42
    )

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True parameters: {true_params.flatten()}")

    # Train model
    hyperparams = ModelHyperparameters(learning_rate=0.1, num_training_iterations=1000)
    model = LinearRegressionModel(hyperparams)
    model.fit(X, y)

    print(f"Learned parameters: {model.weights.flatten()}")
    print(f"R² score: {model.calculate_r2_score(X, y):.4f}")

    # Create a grid for visualization
    x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 50)
    x2_range = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 50)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)

    # Predict on grid
    X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    y_grid = model.predict(X_grid).reshape(X1_grid.shape)

    # True surface
    y_true_grid = (X_grid @ true_params[1:] + true_params[0]).reshape(X1_grid.shape)

    # Create 3D plots
    fig = plt.figure(figsize=(18, 5))

    # Plot 1: Learned surface with data points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X1_grid, X2_grid, y_grid, alpha=0.3, cmap='viridis')
    ax1.scatter(X[:, 0], X[:, 1], y, c='red', marker='o', s=20, alpha=0.6)
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Target')
    ax1.set_title('Learned Linear Regression Surface')

    # Plot 2: True surface with data points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X1_grid, X2_grid, y_true_grid, alpha=0.3, cmap='plasma')
    ax2.scatter(X[:, 0], X[:, 1], y, c='red', marker='o', s=20, alpha=0.6)
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Target')
    ax2.set_title('True Linear Relationship')

    # Plot 3: Residuals in 2D
    ax3 = fig.add_subplot(133)
    y_pred = model.predict(X)
    residuals = (y - y_pred).flatten()
    scatter = ax3.scatter(X[:, 0], X[:, 1], c=residuals, cmap='RdBu', s=50, alpha=0.7)
    plt.colorbar(scatter, ax=ax3, label='Residual')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    ax3.set_title('Residuals in Feature Space')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/linear_regression/linear_regression_2d.png', dpi=150, bbox_inches='tight')
    print("Saved: tests/linear_regression/linear_regression_2d.png")


def plot_polynomial_data():
    """Show that linear regression struggles with polynomial data (without feature engineering)."""
    print("\n" + "="*70)
    print("Linear Regression on Polynomial Data (Demonstrating Limitations)")
    print("="*70)

    # Generate polynomial data
    X, y = generate_polynomial_regression_data(
        n_samples=150,
        degree=2,
        noise=2.0,
        random_state=42
    )

    print(f"\nDataset: {X.shape[0]} samples (quadratic relationship)")

    # Train linear model (will underfit)
    hyperparams = ModelHyperparameters(learning_rate=0.1, epochs=500, regularizer=None, training_method="gradient_descent")
    model = LinearRegressionModel(hyperparams)
    model.fit(X, y)

    r2 = model.calculate_r2_score(X, y)
    print(f"Linear model R² score: {r2:.4f}")

    # Sort for plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    # Predictions
    y_pred = model.predict(X_sorted)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(X, y, alpha=0.6, s=30, label='Data (quadratic)')
    ax.plot(X_sorted, y_pred, 'r-', lw=2, label='Linear fit (underfit)')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Linear Model on Polynomial Data (Underfitting)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('tests/linear_regression/linear_regression_polynomial.png', dpi=150, bbox_inches='tight')
    print("Saved: tests/linear_regression/linear_regression_polynomial.png")
    print("\nNote: Linear model underfits polynomial data. Feature engineering")
    print("      (adding X², X³, etc.) would be needed for a better fit.")


if __name__ == "__main__":
    print("="*70)
    print("Linear Regression Visualization Suite")
    print("="*70)

    # Run all visualizations
    plot_1d_regression()
    plot_2d_regression()
    plot_polynomial_data()

    print("\n" + "="*70)
    print("✅ All visualizations completed successfully!")
    print("="*70)

