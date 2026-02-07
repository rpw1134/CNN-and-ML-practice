from typing import Tuple, Callable

import numpy as np
from numpy.typing import NDArray

def build(X: NDArray, y: NDArray) -> Tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]:
    """
        Builds the mean squared error loss function and its gradient function for a given dataset.
    :param X: np.NDArray: The input data for which the loss function and gradient will be built. It is expected to be a 2D array where rows are samples and columns are features.
    :param y: np.NDArray: The target values corresponding to the input data. It is expected to be a 1D array where each element corresponds to the target value of the respective sample in X.
    :return: Tuple[Callable[[np.NDArray], float], Callable[[np.NDArray], np.NDArray]]:
        A tuple containing the loss function and the gradient function.
        The loss function takes a parameter array as input and returns a float representing the mean squared error.
        The gradient function takes a parameter array as input and returns an array representing the gradient of the loss function with respect to the parameters.
    """

    def loss(parameters: NDArray) -> float:
        predictions = X @ parameters
        return float(0.5 * np.mean((predictions - y) ** 2))

    def gradient(parameters: NDArray) -> NDArray:
        # 1/n X.T @ (y_pred - y) if X is points*features and y is points*1
        predictions = X @ parameters
        grad = (1/X.shape[0]) * X.T @ (predictions - y)
        return grad

    return loss, gradient