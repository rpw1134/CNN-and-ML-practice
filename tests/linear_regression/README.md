# Linear Regression Tests

This directory contains tests and visualizations for the LinearRegressionModel.

## Files

### `test_linear_regression_model.py`
Comprehensive test script that:
- Generates synthetic linear regression data with known parameters
- Trains models with and without L2 regularization
- Compares learned parameters to true parameters
- Evaluates RMSE, MAE, and R² scores
- Creates 6-panel visualization showing model performance

**Run:**
```bash
python tests/linear_regression/test_linear_regression_model.py
```

**Output:**
- Console: Detailed metrics and parameter comparisons
- Plot: `linear_regression_results.png` (6-panel visualization)

### `visualize_linear_regression.py`
Visualization suite demonstrating linear regression in different scenarios:

1. **1D Linear Regression** - Simple line fitting with residual plot
2. **2D Linear Regression** - 3D surface visualization
3. **Polynomial Data** - Shows underfitting when model is too simple

**Run:**
```bash
python tests/linear_regression/visualize_linear_regression.py
```

**Outputs:**
- `linear_regression_1d.png` - 1D regression fit and residuals
- `linear_regression_2d.png` - 3D surface plots and residual heatmap
- `linear_regression_polynomial.png` - Linear model on polynomial data

## Expected Results

### Test Performance Metrics
- **R² Score**: > 0.99 (excellent fit)
- **RMSE**: < 1.0 (low prediction error)
- **Parameter Recovery**: Learned parameters very close to true values

### Visualization Quality
- Clear separation between predicted and true values
- Random residual patterns (indicates good fit)
- L2 regularization shows slight weight shrinkage

## Usage Example

```python
from ml_practice_and_neural_net.ml_practice.models import LinearRegressionModel
from ml_practice_and_neural_net.ml_practice.data_management.dataset_generation import generate_linear_regression_data

# Generate data
X, y, true_params = generate_linear_regression_data(n_samples=200, n_features=3)

# Train model
model = LinearRegressionModel(learning_rate=0.1, num_training_iterations=1000)
model.fit(X, y)

# Evaluate
r2 = model.calculate_r2_score(X, y)
print(f"R² Score: {r2:.4f}")
```

## Notes

- All tests include both unregularized and L2-regularized models
- Visualizations are saved automatically to this directory
- Tests verify that model correctly recovers known parameters
- R² score is used as the primary regression metric

