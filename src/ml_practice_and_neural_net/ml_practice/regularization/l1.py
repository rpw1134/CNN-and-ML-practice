import numpy as np
from numpy.typing import NDArray

def build(lamda_hyperparameter: float):
    def loss(parameters: NDArray) -> float:
        # Skip first element (1D) or first column (2D) - bias term
        if parameters.ndim == 1:
            params_to_reg = parameters[1:]
        else:
            params_to_reg = parameters[:, 1:]
        return lamda_hyperparameter * np.sum(np.abs(params_to_reg))

    def gradient(parameters: NDArray) -> NDArray:
        grad = lamda_hyperparameter * np.sign(parameters)
        # Zero out bias gradient - first element (1D) or first column (2D)
        if parameters.ndim == 1:
            grad[0] = 0
        else:
            grad[:, 0] = 0
        return grad

    return loss, gradient