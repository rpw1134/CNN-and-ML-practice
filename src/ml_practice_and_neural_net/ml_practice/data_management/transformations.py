import numpy as np

def logistic(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))