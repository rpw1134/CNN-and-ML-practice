from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray


def build() -> Tuple[Callable, Callable]:
    def activate(x: NDArray) -> NDArray:
        """max(0, x). Output range: [0, inf)."""
        return np.maximum(0, x)

    def activation_derivative(x: NDArray) -> NDArray:
        """1 where x > 0, else 0."""
        return np.where(x > 0, 1, 0)

    return activate, activation_derivative
