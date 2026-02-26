from numpy.typing import NDArray
from typing import Tuple
import numpy as np


def _softmax(x: NDArray) -> NDArray:
    shifted = x - np.max(x, axis=-1, keepdims=True)  # numerical stability
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def loss(predictions: NDArray, targets: NDArray) -> float:
    """CCE loss. predictions are raw logits, targets are one-hot."""
    probs = np.clip(_softmax(predictions), 1e-15, 1.0)
    return float(-np.mean(np.sum(targets * np.log(probs), axis=1)))


def gradient(predictions: NDArray, targets: NDArray) -> NDArray:
    """dL/dA = (softmax(A) - Y) / batch_size."""
    return _softmax(predictions) - targets


def loss_and_grad(predictions: NDArray, targets: NDArray) -> Tuple[float, NDArray]:
    return loss(predictions, targets), gradient(predictions, targets)
