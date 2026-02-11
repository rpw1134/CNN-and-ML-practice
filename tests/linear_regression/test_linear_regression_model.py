"""
Test script for LinearRegressionModel on synthetic linear regression data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_linear_regression_data
from ml_practice_and_neural_net.ml_practice.models.LinearRegressionModel import LinearRegressionModel


def calculate_rmse(y_true, y_pred):
    """Calculate root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def calculate_mae(y_true, y_pred):
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))


if __name__ == "__main__":
    print("="*70)
    print("Testing LinearRegressionModel on Synthetic Linear Regression Data")
    print("="*70)

    # Generate dataset
    print("\n1. Generating dataset...")
    X, y, true_params = generate_linear_regression_data(
        n_samples=400,
        n_features=3,
        noise=0.5,
        true_coef_range=(-5.0, 5.0),
        true_intercept_range=(-10.0, 10.0),
        random_state=42
    )

    print(f"   Dataset shape: X={X.shape}, y={y.shape}")
    print(f"   Target statistics:")
    print(f"     Mean: {np.mean(y):.3f}")
    print(f"     Std:  {np.std(y):.3f}")
    print(f"     Min:  {np.min(y):.3f}")
    print(f"     Max:  {np.max(y):.3f}")
    print(f"\n   True parameters (including intercept):")
    print(f"     {true_params.flatten()}")

    # Train the model WITHOUT regularization
    print("\n2. Training LinearRegressionModel (no regularization)...")
    model = LinearRegressionModel(
        learning_rate=0.1,
        num_training_iterations=1000
    )
    model.fit(X, y)
    print("   Training complete!")
    print(f"\n   Learned parameters (including intercept):")
    print(f"     {model.weights.flatten()}")

    # Compare learned vs true parameters
    print(f"\n   Parameter comparison:")
    print(f"     {'Component':<15} {'True':<12} {'Learned':<12} {'Difference':<12}")
    print(f"     {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    print(f"     {'Intercept':<15} {true_params[0,0]:>11.4f} {model.weights[0,0]:>11.4f} {abs(true_params[0,0] - model.weights[0,0]):>11.4f}")
    for i in range(1, len(true_params)):
        print(f"     {'Weight ' + str(i):<15} {true_params[i,0]:>11.4f} {model.weights[i,0]:>11.4f} {abs(true_params[i,0] - model.weights[i,0]):>11.4f}")

    # Evaluate on full dataset
    print("\n3. Evaluating model...")
    y_pred = model.predict(X)

    rmse = calculate_rmse(y, y_pred)
    mae = calculate_mae(y, y_pred)
    r2 = model.calculate_r2_score(X, y)

    print(f"   Full dataset metrics:")
    print(f"     RMSE:       {rmse:.6f}")
    print(f"     MAE:        {mae:.6f}")
    print(f"     R² score:   {r2:.6f}")

    # Loss values
    print("\n4. Loss values (MSE):")
    train_loss = model.evaluate_mse_training_loss()
    test_loss = model.evaluate_mse_testing_loss()
    print(f"   Training loss: {train_loss:.6f}")
    print(f"   Testing loss:  {test_loss:.6f}")

    # Train model WITH L2 regularization
    print("\n5. Training LinearRegressionModel with L2 regularization...")
    model_l2 = LinearRegressionModel(
        learning_rate=0.1,
        num_training_iterations=1000,
        reg_technique="l2",
        lambda_reg=0.1
    )
    model_l2.fit(X, y)
    print("   Training complete!")
    print(f"\n   Learned parameters (with L2):")
    print(f"     {model_l2.weights.flatten()}")

    y_pred_l2 = model_l2.predict(X)
    rmse_l2 = calculate_rmse(y, y_pred_l2)
    r2_l2 = model_l2.calculate_r2_score(X, y)

    print(f"\n   L2-regularized model metrics:")
    print(f"     RMSE:       {rmse_l2:.6f}")
    print(f"     R² score:   {r2_l2:.6f}")
    print(f"     Weight magnitude: {np.linalg.norm(model_l2.weights[1:]):.6f}")
    print(f"   vs. non-regularized weight magnitude: {np.linalg.norm(model.weights[1:]):.6f}")

    # Create visualizations
    print("\n6. Creating visualizations...")

    # For 3 features, we'll create multiple plots
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: Predictions vs True Values (scatter)
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y, y_pred, alpha=0.5, s=20, label='No regularization')
    ax1.scatter(y, y_pred_l2, alpha=0.5, s=20, label='L2 regularization')
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Perfect fit')
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predictions vs True Values')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals (no regularization)
    ax2 = plt.subplot(2, 3, 2)
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot (No Regularization)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residuals (L2 regularization)
    ax3 = plt.subplot(2, 3, 3)
    residuals_l2 = y - y_pred_l2
    ax3.scatter(y_pred_l2, residuals_l2, alpha=0.5, s=20, color='orange')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Plot (L2 Regularization)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature 1 vs Target
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(X[:, 0], y, alpha=0.5, s=20, label='True data')
    # Create a range for plotting the fit line
    x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).reshape(-1, 1)
    # For visualization, fix other features at their mean
    X_plot = np.column_stack([x1_range,
                               np.full_like(x1_range, X[:, 1].mean()),
                               np.full_like(x1_range, X[:, 2].mean())])
    y_plot = model.predict(X_plot)
    ax4.plot(x1_range, y_plot, 'r-', lw=2, label='Model fit')
    ax4.set_xlabel('Feature 1')
    ax4.set_ylabel('Target')
    ax4.set_title('Feature 1 vs Target (other features at mean)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Parameter comparison
    ax5 = plt.subplot(2, 3, 5)
    param_names = ['Intercept'] + [f'Weight {i}' for i in range(1, len(true_params))]
    x_pos = np.arange(len(param_names))
    width = 0.25
    ax5.bar(x_pos - width, true_params.flatten(), width, label='True', alpha=0.8)
    ax5.bar(x_pos, model.weights.flatten(), width, label='Learned (no reg)', alpha=0.8)
    ax5.bar(x_pos + width, model_l2.weights.flatten(), width, label='Learned (L2)', alpha=0.8)
    ax5.set_xlabel('Parameter')
    ax5.set_ylabel('Value')
    ax5.set_title('Parameter Comparison')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(param_names, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Error distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(residuals, bins=30, alpha=0.5, label='No regularization', density=True)
    ax6.hist(residuals_l2, bins=30, alpha=0.5, label='L2 regularization', density=True)
    ax6.axvline(x=0, color='r', linestyle='--', lw=2)
    ax6.set_xlabel('Residual Value')
    ax6.set_ylabel('Density')
    ax6.set_title('Residual Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tests/linear_regression/linear_regression_results.png', dpi=150, bbox_inches='tight')
    print("   Saved plot to: tests/linear_regression/linear_regression_results.png")

    print("\n" + "="*70)
    if r2 >= 0.95:
        print("✅ SUCCESS: Model achieved excellent R² score (>0.95)!")
    elif r2 >= 0.85:
        print("⚠️  GOOD: Model achieved good R² score (>0.85).")
    else:
        print("❌ ISSUE: Model R² score is lower than expected.")

    if rmse < 1.0:
        print("✅ SUCCESS: RMSE is low (<1.0)!")
    elif rmse < 2.0:
        print("⚠️  GOOD: RMSE is acceptable (<2.0).")
    else:
        print("❌ ISSUE: RMSE is higher than expected.")
    print("="*70)

