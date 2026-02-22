from typing import Tuple, Callable

import numpy as np
from numpy.typing import NDArray
def build() -> Tuple[Callable, Callable]:

    def activate(x: NDArray)->NDArray:
        return 1/ (1+np.exp(-x))

    def activation_derivative(x: NDArray)->NDArray:
        return activate(x)*(1-activate(x))

    return activate, activation_derivative
