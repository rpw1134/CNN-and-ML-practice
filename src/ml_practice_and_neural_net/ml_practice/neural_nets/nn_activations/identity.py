from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray


def build() -> Tuple[Callable, Callable]:
    def activate(x: NDArray) -> NDArray:
        """f(x) = x. Used in output layers for regression."""
        return x

    def activation_derivative(x: NDArray) -> NDArray:
        """f'(x) = 1 everywhere."""
        return np.ones_like(x)

    return activate, activation_derivative

