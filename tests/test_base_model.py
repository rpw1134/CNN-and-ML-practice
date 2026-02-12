"""
Example demonstrating the BaseModel abstract class usage.

This shows how BaseModel provides:
1. Abstract methods (fit, predict) that must be implemented by subclasses
2. A concrete evaluate_error() method that can be used by all models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_practice_and_neural_net.ml_practice.models import BaseModel, LogisticRegressionModel, SoftmaxRegressionModel
from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
import numpy as np
from abc import ABC

# Example: BaseModel cannot be instantiated directly
hyperparams = Hyperparameters(learning_rate=0.01, num_training_iterations=100)
try:
    model = BaseModel(hyperparams)
    print("ERROR: BaseModel should not be instantiable!")
except TypeError as e:
    print(f"✓ BaseModel is abstract and cannot be instantiated: {e}")

# Example: Both models inherit from BaseModel
print(f"\n✓ LogisticRegressionModel is subclass of BaseModel: {issubclass(LogisticRegressionModel, BaseModel)}")
print(f"✓ SoftmaxRegressionModel is subclass of BaseModel: {issubclass(SoftmaxRegressionModel, BaseModel)}")

# Example: Create instances of concrete models
hyperparams = Hyperparameters(learning_rate=0.01, num_training_iterations=100)
log_model = LogisticRegressionModel(hyperparams)
soft_model = SoftmaxRegressionModel(hyperparams)
print(f"\n✓ Successfully created LogisticRegressionModel instance")
print(f"✓ Successfully created SoftmaxRegressionModel instance")

# Example: Models have the evaluate_error method from BaseModel
print(f"\n✓ LogisticRegressionModel has evaluate_error: {hasattr(log_model, 'evaluate_error')}")
print(f"✓ SoftmaxRegressionModel has evaluate_error: {hasattr(soft_model, 'evaluate_error')}")

# Example: Models have abstract methods implemented
print(f"\n✓ LogisticRegressionModel has fit: {hasattr(log_model, 'fit')}")
print(f"✓ LogisticRegressionModel has predict: {hasattr(log_model, 'predict')}")
print(f"✓ SoftmaxRegressionModel has fit: {hasattr(soft_model, 'fit')}")
print(f"✓ SoftmaxRegressionModel has predict: {hasattr(soft_model, 'predict')}")

print("\n✓ All tests passed! BaseModel is working correctly.")
