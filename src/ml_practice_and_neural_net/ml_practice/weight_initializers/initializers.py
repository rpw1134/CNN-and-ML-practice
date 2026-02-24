from numpy.typing import NDArray
import numpy as np

def normal(input_dim: int, output_dim: int) -> NDArray:
    """Initialize weights with a normal distribution, scaled by the square root of the input dimension."""
    stddev = np.sqrt(1.0 / input_dim)
    return np.random.normal(0, stddev, (output_dim, input_dim))

def uniform(input_dim: int, output_dim: int) -> NDArray:
    """Initialize weights with a uniform distribution, scaled by the square root of the input dimension."""
    limit = np.sqrt(1.0 / input_dim)
    return np.random.uniform(-limit, limit, (output_dim, input_dim))

def he_normal(input_dim: int, output_dim: int) -> NDArray:
    """He initialization for ReLU activations."""
    stddev = np.sqrt(2.0 / input_dim)
    return np.random.normal(0, stddev, (output_dim, input_dim))

def he_uniform(input_dim: int, output_dim: int) -> NDArray:
    """He initialization for ReLU activations."""
    limit = np.sqrt(6.0 / input_dim)
    return np.random.uniform(-limit, limit, (output_dim, input_dim))

def xavier_normal(input_dim: int, output_dim: int) -> NDArray:
    """Xavier initialization for sigmoid/tanh activations."""
    stddev = np.sqrt(2.0 / (input_dim + output_dim))
    return np.random.normal(0, stddev, (output_dim, input_dim))

def xavier_uniform(input_dim: int, output_dim: int) -> NDArray:
    """Xavier initialization for sigmoid/tanh activations."""
    limit = np.sqrt(6.0 / (input_dim + output_dim))
    return np.random.uniform(-limit, limit, (output_dim, input_dim))
