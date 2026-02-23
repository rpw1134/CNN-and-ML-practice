from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray


def build() -> Tuple[Callable, Callable]:
    def activate(x: NDArray, alpha: float = 0.01) -> NDArray:
        """x where x > 0, alpha*x otherwise. Prevents dying neurons vs. ReLU."""
        return np.maximum(alpha * x, x)

    def activation_derivative(x: NDArray, alpha: float = 0.01) -> NDArray:
        """1 where x > 0, else alpha."""
        return np.where(x > 0, 1, alpha)

    return activate, activation_derivative
