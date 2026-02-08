import numpy as np
from numpy.typing import NDArray

from ml_practice_and_neural_net.ml_practice.data_management.transformations import softmax


def build(X: NDArray, Y: NDArray):
    """
        Builds the cross-entropy loss function and its gradient function for a given dataset.
    :param X: np.NDArray: The input data for which the loss function and gradient will be built. NxD
    :param Y: np.NDArray: The target values corresponding to the input data. NxC one hot
    :return: Tuple[Callable[[np.NDArray], float], Callable[[np.NDArray], np.NDArray]]:
        A tuple containing the loss function and the gradient function.
        The loss function takes a parameter array as input and returns a float representing the mean squared error.
        The gradient function takes a parameter array as input and returns an array representing the gradient of the loss function with respect to the parameters.
    """

    def loss(parameters: NDArray) -> float:
        logits = X @ parameters.T # NxC
        one_hots = np.sum(logits * Y, axis=1) # NxC @ CxN gives NxN, summed right gives Nx1
        log_sum_exp = np.log(np.sum(np.exp(logits), axis=1)) # NxC gives Nx1
        return -1 * float(np.mean(log_sum_exp - one_hots))

    def gradient(parameters: NDArray) -> NDArray:
        # find prediction vectors for each x, subtract the corresponding 1 hot label to get gradient wrt logits
        # then logit wrt parameters is just x, so we can do x.T @ (preds - one_hots) to get the gradient wrt parameters
        logits = X @ parameters.T # NxC
        softmax_outputs = softmax(logits) # NxC
        grad = (softmax_outputs - Y).T @ X # CxN @ NxD gives CxD
        return grad / X.shape[0] # average to keep cost function consistent

    return loss, gradient

