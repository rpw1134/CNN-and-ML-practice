from typing import Tuple, Callable

import numpy as np
from numpy.typing import NDArray

def build() -> Tuple[Callable, Callable]:

    def activate(x: NDArray, alpha: float = 0.01) -> NDArray:
        return np.maximum(alpha * x, x)

    def activation_derivative(x: NDArray, alpha: float = 0.01) -> NDArray:
        return np.where(x > 0, 1, alpha)

    return activate, activation_derivative