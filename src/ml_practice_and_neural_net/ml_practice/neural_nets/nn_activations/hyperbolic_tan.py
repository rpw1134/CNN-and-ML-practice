from typing import Tuple, Callable
import numpy as np
from numpy.typing import NDArray

def build() -> Tuple[Callable, Callable]:

    def activate(x: NDArray) -> NDArray:
        return np.tanh(x)

    def activation_derivative(x: NDArray) -> NDArray:
        return 1 - np.tanh(x) ** 2

    return activate, activation_derivative

