from numpy.typing import NDArray
from typing import Tuple
import numpy as np

def loss(predictions: NDArray, targets: NDArray) -> float:
    return float(np.mean(1/2 * ((predictions - targets) ** 2)))

def gradient(predictions: NDArray, targets: NDArray) -> NDArray:
    return predictions - targets  # (batch_size, output_dim), per-sample gradients

def loss_and_grad(predictions: NDArray, targets: NDArray) -> Tuple[float, NDArray]:
    return loss(predictions, targets), gradient(predictions, targets)
