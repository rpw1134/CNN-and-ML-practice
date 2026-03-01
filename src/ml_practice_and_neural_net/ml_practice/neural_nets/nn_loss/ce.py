from numpy.typing import NDArray
import numpy as np
from ml_practice_and_neural_net.ml_practice.data_management.transformations import logistic


def loss(predictions: NDArray, targets: NDArray) -> float:
    """Binary cross-entropy. predictions are raw logits."""
    probs = np.clip(logistic(predictions), 1e-15, 1 - 1e-15)
    return float(np.mean(-targets * np.log(probs) - (1 - targets) * np.log(1 - probs)))


def gradient(predictions: NDArray, targets: NDArray) -> NDArray:
    """dL/dA per sample = sigmoid(A) - y, shape: (batch_size, 1).
    Batch averaging is handled by the layer, not here."""
    return logistic(predictions) - targets
