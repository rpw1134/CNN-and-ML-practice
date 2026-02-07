"""
Simple linearly separable dataset generator for testing logistic regression.
"""
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def generate_simple_linear_data(n_samples: int = 200,
                                n_features: int = 2,
                                separation: float = 2.0,
                                noise: float = 0.8,
                                random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Generate a simple linearly separable binary classification dataset.

    Parameters:
    -----------
    n_samples : int
        Total number of samples (will be split evenly between classes)
    n_features : int
        Number of features (dimensions)
    separation : float
        Distance between cluster centers (larger = more separated)
    noise : float
        Standard deviation of the Gaussian noise (larger = more overlap)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : NDArray of shape (n_samples, n_features)
        Feature matrix
    y : NDArray of shape (n_samples, 1)
        Binary labels (0 or 1)

    Example:
    --------
    >>> X, y = generate_simple_linear_data(n_samples=100, n_features=2)
    >>> print(X.shape, y.shape)
    (100, 2) (100, 1)
    """
    np.random.seed(random_state)

    half = n_samples // 2

    # Class 0: centered around negative values
    center_0 = -separation * np.ones(n_features)
    X_class_0 = np.random.randn(half, n_features) * noise + center_0

    # Class 1: centered around positive values
    center_1 = separation * np.ones(n_features)
    X_class_1 = np.random.randn(half, n_features) * noise + center_1

    # Combine
    X = np.vstack([X_class_0, X_class_1])
    y = np.vstack([np.zeros((half, 1)), np.ones((half, 1))])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def generate_xor_data(n_samples: int = 200,
                      noise: float = 0.3,
                      random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Generate an XOR dataset (NOT linearly separable - for testing limitations).

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    noise : float
        Standard deviation of the Gaussian noise
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : NDArray of shape (n_samples, 2)
        Feature matrix
    y : NDArray of shape (n_samples, 1)
        Binary labels (0 or 1)
    """
    np.random.seed(random_state)

    quarter = n_samples // 4

    # Class 0: top-left and bottom-right quadrants
    X_0_tl = np.random.randn(quarter, 2) * noise + np.array([-1, 1])
    X_0_br = np.random.randn(quarter, 2) * noise + np.array([1, -1])

    # Class 1: top-right and bottom-left quadrants
    X_1_tr = np.random.randn(quarter, 2) * noise + np.array([1, 1])
    X_1_bl = np.random.randn(quarter, 2) * noise + np.array([-1, -1])

    # Combine
    X = np.vstack([X_0_tl, X_0_br, X_1_tr, X_1_bl])
    y = np.vstack([
        np.zeros((quarter * 2, 1)),
        np.ones((quarter * 2, 1))
    ])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def generate_circle_data(n_samples: int = 200,
                        inner_radius: float = 1.0,
                        outer_radius: float = 3.0,
                        noise: float = 0.2,
                        random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Generate concentric circles dataset (NOT linearly separable).

    Parameters:
    -----------
    n_samples : int
        Total number of samples
    inner_radius : float
        Radius of inner circle (class 0)
    outer_radius : float
        Radius of outer circle (class 1)
    noise : float
        Standard deviation of the radial noise
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : NDArray of shape (n_samples, 2)
        Feature matrix
    y : NDArray of shape (n_samples, 1)
        Binary labels (0 or 1)
    """
    np.random.seed(random_state)

    half = n_samples // 2

    # Inner circle (class 0)
    angles_0 = np.random.uniform(0, 2 * np.pi, half)
    radii_0 = np.abs(np.random.randn(half) * noise + inner_radius)
    X_0 = np.column_stack([radii_0 * np.cos(angles_0),
                           radii_0 * np.sin(angles_0)])

    # Outer circle (class 1)
    angles_1 = np.random.uniform(0, 2 * np.pi, half)
    radii_1 = np.abs(np.random.randn(half) * noise + outer_radius)
    X_1 = np.column_stack([radii_1 * np.cos(angles_1),
                           radii_1 * np.sin(angles_1)])

    # Combine
    X = np.vstack([X_0, X_1])
    y = np.vstack([np.zeros((half, 1)), np.ones((half, 1))])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":
    # Demo usage
    print("Generating sample datasets...\n")

    # Linear data
    X_linear, y_linear = generate_simple_linear_data(n_samples=200)
    print(f"Linear dataset: X shape={X_linear.shape}, y shape={y_linear.shape}")
    print(f"  Class 0: {np.sum(y_linear == 0)} samples")
    print(f"  Class 1: {np.sum(y_linear == 1)} samples\n")

    # XOR data
    X_xor, y_xor = generate_xor_data(n_samples=200)
    print(f"XOR dataset: X shape={X_xor.shape}, y shape={y_xor.shape}")
    print(f"  Class 0: {np.sum(y_xor == 0)} samples")
    print(f"  Class 1: {np.sum(y_xor == 1)} samples\n")

    # Circle data
    X_circle, y_circle = generate_circle_data(n_samples=200)
    print(f"Circle dataset: X shape={X_circle.shape}, y shape={y_circle.shape}")
    print(f"  Class 0: {np.sum(y_circle == 0)} samples")
    print(f"  Class 1: {np.sum(y_circle == 1)} samples")

