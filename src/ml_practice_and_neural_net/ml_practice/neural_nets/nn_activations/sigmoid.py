from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray


def build() -> Tuple[Callable, Callable]:
    def activate(x: NDArray) -> NDArray:
        """1 / (1 + exp(-x)). Output range: (0, 1)."""
        return 1 / (1 + np.exp(-x))

    def activation_derivative(x: NDArray) -> NDArray:
        """sigmoid(x) * (1 - sigmoid(x)). Output range: (0, 0.25]."""
        return activate(x) * (1 - activate(x))

    return activate, activation_derivative
