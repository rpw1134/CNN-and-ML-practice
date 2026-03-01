from numpy.typing import NDArray
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
    """dL/dA per sample = softmax(A) - Y, shape: (batch_size, num_classes).
    Batch averaging is handled by the layer, not here."""
    return _softmax(predictions) - targets
