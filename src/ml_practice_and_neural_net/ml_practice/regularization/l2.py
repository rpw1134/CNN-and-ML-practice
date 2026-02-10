import numpy as np
from numpy.typing import NDArray

def build(lambda_hyperparameter: float):

    def loss(parameters: NDArray) -> float:
        # Skip first element (1D) or first column (2D) - bias term
        if parameters.ndim == 1:
            params_to_reg = parameters[1:]
        else:
            params_to_reg = parameters[:, 1:]
        return 0.5 * lambda_hyperparameter * np.sum(params_to_reg ** 2)

    def gradient(parameters: NDArray) -> NDArray:
        grad = lambda_hyperparameter * parameters
        # Zero out bias gradient - first element (1D) or first column (2D)
        if parameters.ndim == 1:
            grad[0] = 0
        else:
            grad[:, 0] = 0
        return grad

    return loss, gradient