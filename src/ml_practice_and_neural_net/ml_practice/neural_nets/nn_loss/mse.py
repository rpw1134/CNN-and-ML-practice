from numpy.typing import NDArray
import numpy as np

def loss(predictions: NDArray, targets: NDArray) -> float:
    return float(np.mean(1/2 * ((predictions - targets) ** 2)))

def gradient(predictions: NDArray, targets: NDArray) -> NDArray:
    """dL/dA per sample = predictions - targets, shape: (batch_size, output_dim).
    Batch averaging is handled by the layer, not here."""
    return predictions - targets  # (batch_size, output_dim), per-sample gradients

