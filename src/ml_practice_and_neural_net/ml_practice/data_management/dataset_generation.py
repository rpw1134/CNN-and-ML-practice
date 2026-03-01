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


def generate_multiclass_linear_data(n_samples: int = 300,
                                    n_features: int = 2,
                                    n_classes: int = 3,
                                    separation: float = 5.0,
                                    noise: float = 0.5,
                                    random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Generate a linearly separable multi-class classification dataset.
    Classes are arranged in a circle pattern in feature space.

    Parameters:
    -----------
    n_samples : int
        Total number of samples (will be split evenly between classes)
    n_features : int
        Number of features (dimensions)
    n_classes : int
        Number of classes
    separation : float
        Distance of cluster centers from origin (larger = more separated)
    noise : float
        Standard deviation of the Gaussian noise (larger = more overlap)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : NDArray of shape (n_samples, n_features)
        Feature matrix
    y : NDArray of shape (n_samples,)
        Class labels (0, 1, 2, ..., n_classes-1)

    Example:
    --------
    >>> X, y = generate_multiclass_linear_data(n_samples=300, n_classes=3)
    >>> print(X.shape, y.shape)
    (300, 2) (300,)
    """
    np.random.seed(random_state)

    samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []

    for class_idx in range(n_classes):
        # Position classes in a circle (for 2D) or evenly distributed in space
        if n_features == 2:
            angle = 2 * np.pi * class_idx / n_classes
            center = separation * np.array([np.cos(angle), np.sin(angle)])
        else:
            # For higher dimensions, create orthogonal-ish centers
            center = np.zeros(n_features)
            center[class_idx % n_features] = separation * (1 if class_idx // n_features % 2 == 0 else -1)
            if n_features > 1:
                center[(class_idx + 1) % n_features] = separation * 0.5 * (class_idx / n_classes - 0.5)

        # Generate samples around the center
        X_class = np.random.randn(samples_per_class, n_features) * noise + center
        y_class = np.full(samples_per_class, class_idx)

        X_list.append(X_class)
        y_list.append(y_class)

    # Combine all classes
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    indices = np.random.permutation(len(y))
    X = X[indices]
    y = y[indices]

    return X, y


def generate_linear_regression_data(n_samples: int = 200,
                                    n_features: int = 3,
                                    noise: float = 0.5,
                                    true_coef_range: tuple = (-5.0, 5.0),
                                    true_intercept_range: tuple = (-10.0, 10.0),
                                    random_state: int = 42) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Generate synthetic data for linear regression with a known linear relationship.

    y = X @ true_weights + true_intercept + noise

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of input features
    noise : float
        Standard deviation of Gaussian noise added to outputs
    true_coef_range : tuple
        (min, max) range for true coefficient values
    true_intercept_range : tuple
        (min, max) range for true intercept value
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : NDArray of shape (n_samples, n_features)
        Feature matrix
    y : NDArray of shape (n_samples, 1)
        Target values (continuous)
    true_params : NDArray of shape (n_features + 1, 1)
        True parameters [intercept, weight_1, ..., weight_n]
        Used to verify model performance

    Example:
    --------
    >>> X, y, true_params = generate_linear_regression_data(n_samples=100, n_features=2)
    >>> print(X.shape, y.shape, true_params.shape)
    (100, 2) (100, 1) (3, 1)
    """
    np.random.seed(random_state)

    # Generate random input features from standard normal distribution
    X = np.random.randn(n_samples, n_features)

    # Generate true weights and intercept
    true_weights = np.random.uniform(true_coef_range[0], true_coef_range[1], (n_features, 1))
    true_intercept = np.random.uniform(true_intercept_range[0], true_intercept_range[1])

    # Generate target values: y = X @ weights + intercept + noise
    y = X @ true_weights + true_intercept + np.random.randn(n_samples, 1) * noise

    # Combine intercept and weights for verification
    true_params = np.vstack([[[true_intercept]], true_weights])

    return X, y, true_params


def generate_polynomial_regression_data(n_samples: int = 200,
                                        degree: int = 2,
                                        noise: float = 1.0,
                                        x_range: tuple = (-3.0, 3.0),
                                        random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Generate synthetic data for polynomial regression (1D input).

    Useful for demonstrating that linear regression can handle polynomial features
    when combined with polynomial feature transformation.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    degree : int
        Degree of the polynomial (e.g., 2 for quadratic)
    noise : float
        Standard deviation of Gaussian noise
    x_range : tuple
        (min, max) range for input feature x
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    X : NDArray of shape (n_samples, 1)
        Feature matrix (single feature)
    y : NDArray of shape (n_samples, 1)
        Target values following a polynomial relationship

    Example:
    --------
    >>> X, y = generate_polynomial_regression_data(n_samples=100, degree=2)
    >>> print(X.shape, y.shape)
    (100, 1) (100, 1)
    """
    np.random.seed(random_state)

    # Generate random x values
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))

    # Generate polynomial coefficients
    coefficients = np.random.uniform(-2, 2, degree + 1)

    # Calculate y = a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0 + noise
    y = np.zeros((n_samples, 1))
    for i, coef in enumerate(coefficients):
        y += coef * (X ** i)

    # Add noise
    y += np.random.randn(n_samples, 1) * noise

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
    print(f"  Class 1: {np.sum(y_circle == 1)} samples\n")

    # Multiclass data
    X_multi, y_multi = generate_multiclass_linear_data(n_samples=300, n_classes=3)
    print(f"Multiclass dataset: X shape={X_multi.shape}, y shape={y_multi.shape}")
    for i in range(3):
        print(f"  Class {i}: {np.sum(y_multi == i)} samples")

    # Linear regression data
    X_reg, y_reg, true_params = generate_linear_regression_data(n_samples=200, n_features=3)
    print(f"\nLinear regression dataset: X shape={X_reg.shape}, y shape={y_reg.shape}")
    print(f"  True parameters shape: {true_params.shape}")
    print(f"  Target mean: {np.mean(y_reg):.2f}, std: {np.std(y_reg):.2f}")

    # Polynomial regression data
    X_poly, y_poly = generate_polynomial_regression_data(n_samples=200, degree=2)
    print(f"\nPolynomial regression dataset: X shape={X_poly.shape}, y shape={y_poly.shape}")
    print(f"  Input range: [{np.min(X_poly):.2f}, {np.max(X_poly):.2f}]")
    print(f"  Target mean: {np.mean(y_poly):.2f}, std: {np.std(y_poly):.2f}")


def generate_circle_data(n_samples: int = 1000, radius: float = 1.0, noise: float = 0.05,
                         random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Balanced binary classification dataset: points inside a circle vs. outside.

    Parameters:
    -----------
    n_samples : int
        Total number of samples (split evenly between inside and outside).
    radius : float
        Radius of the circle boundary.
    noise : float
        Standard deviation of Gaussian noise added to each point.
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    X : NDArray of shape (n_samples, 2)
        2D feature matrix.
    y : NDArray of shape (n_samples,)
        Binary labels — 1 = inside, 0 = outside.
    """
    rng = np.random.RandomState(random_state)
    half = n_samples // 2

    # inside: uniform area sampling in polar coords
    r_in = rng.uniform(0, radius, half) ** 0.5 * radius
    theta_in = rng.uniform(0, 2 * np.pi, half)
    X_in = np.c_[r_in * np.cos(theta_in), r_in * np.sin(theta_in)]
    X_in += rng.normal(0, noise, X_in.shape)

    # outside: annulus from radius to 2*radius
    r_out = rng.uniform(radius, 2.0 * radius, half) ** 0.5 * np.sqrt(2) * radius
    r_out = np.clip(r_out, radius + 0.05, 2.5)
    theta_out = rng.uniform(0, 2 * np.pi, half)
    X_out = np.c_[r_out * np.cos(theta_out), r_out * np.sin(theta_out)]
    X_out += rng.normal(0, noise, X_out.shape)

    X = np.vstack([X_in, X_out])
    y = np.array([1] * half + [0] * half)
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def generate_sine_data(n_samples: int = 800, noise: float = 0.05,
                       random_state: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Regression dataset: y = sin(x) + noise, x uniformly sampled from [-π, π].

    Parameters:
    -----------
    n_samples : int
        Number of samples.
    noise : float
        Standard deviation of Gaussian noise added to y.
    random_state : int
        Random seed for reproducibility.

    Returns:
    --------
    X : NDArray of shape (n_samples, 1)
        Input values.
    y : NDArray of shape (n_samples, 1)
        Noisy sine targets.
    """
    rng = np.random.RandomState(random_state)
    x = rng.uniform(-np.pi, np.pi, n_samples)
    y = np.sin(x) + rng.normal(0, noise, n_samples)
    return x.reshape(-1, 1), y.reshape(-1, 1)
