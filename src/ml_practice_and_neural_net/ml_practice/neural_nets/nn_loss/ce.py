from numpy.typing import NDArray
from typing import Tuple
import numpy as np


def _sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-x))


def loss(predictions: NDArray, targets: NDArray) -> float:
    """Binary cross-entropy. predictions are raw logits."""
    probs = np.clip(_sigmoid(predictions), 1e-15, 1 - 1e-15)
    return float(np.mean(-targets * np.log(probs) - (1 - targets) * np.log(1 - probs)))


def gradient(predictions: NDArray, targets: NDArray) -> NDArray:
    """dL/dA = (sigmoid(A) - y) / batch_size."""
    return _sigmoid(predictions) - targets


def loss_and_grad(predictions: NDArray, targets: NDArray) -> Tuple[float, NDArray]:
    return loss(predictions, targets), gradient(predictions, targets)
