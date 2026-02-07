# Logistic Regression Test Datasets

This directory contains utilities for generating test datasets for your logistic regression model.

## Quick Start

```python
from ml_practice.datasets import generate_simple_linear_data
from ml_practice.models.LogisticRegressionModel import LogisticRegressionModel

# Generate linearly separable data
X, y = generate_simple_linear_data(n_samples=200, n_features=2)

# Train model
model = LogisticRegressionModel(learning_rate=0.1, num_training_iterations=1000)
model.fit(X, y)

# Evaluate
print(f"Training Loss: {model.evaluate_ce_training_loss():.4f}")
print(f"Testing Loss: {model.evaluate_ce_testing_loss():.4f}")
```

## Available Datasets

### 1. `generate_simple_linear_data()` - **LINEARLY SEPARABLE** ✅
Generates two Gaussian clusters that are linearly separable.

**Parameters:**
- `n_samples` (int): Total number of samples (default: 200)
- `n_features` (int): Number of features (default: 2)
- `separation` (float): Distance between clusters (default: 2.0)
- `noise` (float): Standard deviation of noise (default: 0.8)
- `random_state` (int): Random seed (default: 42)

**Example:**
```python
X, y = generate_simple_linear_data(n_samples=200, separation=3.0, noise=0.5)
# Higher separation + lower noise = easier to separate
```

### 2. `generate_xor_data()` - **NOT LINEARLY SEPARABLE** ❌
Generates XOR pattern data (logistic regression will struggle with this).

**Parameters:**
- `n_samples` (int): Total number of samples (default: 200)
- `noise` (float): Standard deviation of noise (default: 0.3)
- `random_state` (int): Random seed (default: 42)

**Example:**
```python
X, y = generate_xor_data(n_samples=200)
# This will show the limitations of linear models
```

### 3. `generate_circle_data()` - **NOT LINEARLY SEPARABLE** ❌
Generates concentric circles (inner = class 0, outer = class 1).

**Parameters:**
- `n_samples` (int): Total number of samples (default: 200)
- `inner_radius` (float): Radius of inner circle (default: 1.0)
- `outer_radius` (float): Radius of outer circle (default: 3.0)
- `noise` (float): Standard deviation of radial noise (default: 0.2)
- `random_state` (int): Random seed (default: 42)

**Example:**
```python
X, y = generate_circle_data(n_samples=200, inner_radius=1.0, outer_radius=3.0)
# Another example where linear classification fails
```

## Test Results

Running the full test with visualization:
```bash
python -m src.ml_practice_and_neural_net.ml_practice.test_logistic_regression
```

Expected output for linearly separable data:
- Training Accuracy: ~100%
- Testing Accuracy: ~100%
- Training Loss: < 0.01
- Testing Loss: < 0.01

## Files

- **`datasets.py`**: Dataset generation functions
- **`test_logistic_regression.py`**: Full test with visualization and metrics
- **`example_usage.py`**: Quick example showing basic usage

## Notes

- All datasets return `X` as shape `(n_samples, n_features)` and `y` as shape `(n_samples, 1)`
- The data is automatically shuffled
- Labels are binary: 0 or 1
- Use linearly separable data to verify your model works correctly
- Use non-linearly separable data to understand model limitations

