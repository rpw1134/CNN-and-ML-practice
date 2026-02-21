# Perceptron Model Tests

This directory contains tests for the Perceptron classification model.

## Overview

The Perceptron is a binary linear classifier that learns a decision boundary to separate two classes. It's one of the simplest neural network models and only works on linearly separable data.

## Test Files

### `test_perceptron.py`
Main test script that:
- Generates a linearly separable binary classification dataset using `generate_simple_linear_data()`
- Converts labels from {0, 1} to {-1, +1} format (required for perceptron)
- Splits data into training (80%) and test (20%) sets
- Trains the perceptron model
- Evaluates performance on both training and test sets
- Visualizes the decision boundary and classified points

## Running the Tests

```bash
cd /Users/ryanwilliams/Projects/ml-practice-and-neural-net
python tests/perceptron/test_perceptron.py
```

## Expected Results

On linearly separable data, the perceptron should achieve:
- **Training Accuracy**: ~100% (since data is linearly separable)
- **Test Accuracy**: ~100% (with proper separation and low noise)

The visualization shows:
- Training data with the learned decision boundary
- Test data with the decision boundary and any misclassified points marked

## Model Details

**Algorithm**: Perceptron Learning Algorithm
- **Update Rule**: For misclassified points: `w = w + η * X^T * y`
- **Prediction**: `sign(X · w)`
- **Convergence**: Guaranteed for linearly separable data

**Hyperparameters Used**:
- Learning rate: 0.1
- Max epochs: 100
- No regularization (perceptron doesn't use regularization)

## Notes

- The perceptron only works on **linearly separable** datasets
- For non-linearly separable data, the algorithm may not converge
- Labels must be in {-1, +1} format (not {0, 1})
- The model adds a bias term automatically during training

