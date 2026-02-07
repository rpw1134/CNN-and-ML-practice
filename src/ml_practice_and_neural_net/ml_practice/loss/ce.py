import numpy as np
from numpy.typing import NDArray
from ..data_management import transformations

def build(X: NDArray, y: NDArray):

    def loss(parameters: NDArray) -> float:
        logits = X @ parameters
        logistics = transformations.logistic(logits)
        logistics = np.clip(logistics, 1e-15, 1-1e-15)
        return float(np.mean(-1*y * np.log(logistics) - (1-y) * np.log(1-logistics)))

    # Predictions use logistic function as well
    def gradient(parameters: NDArray) -> NDArray:
        logits = X @ parameters
        logistics = transformations.logistic(logits)
        grad = (1/X.shape[0]) * X.T @ (logistics - y)
        return grad

    return loss, gradient