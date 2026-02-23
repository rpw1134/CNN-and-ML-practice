from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray


def build() -> Tuple[Callable, Callable]:
    def activate(x: NDArray) -> NDArray:
        """tanh(x). Output range: (-1, 1)."""
        return np.tanh(x)

    def activation_derivative(x: NDArray) -> NDArray:
        """1 - tanh(x)^2. Output range: (0, 1]."""
        return 1 - np.tanh(x) ** 2

    return activate, activation_derivative
