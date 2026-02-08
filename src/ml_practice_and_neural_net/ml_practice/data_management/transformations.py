import numpy as np

def logistic(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-logits))

def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)