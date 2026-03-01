from dataclasses import dataclass

@dataclass
class NNHyperparameters:
    """
    A data class to encapsulate all necessary hyperparameters required for training a neural network.
    This includes the learning rate, number of training iterations, regularization strength, and batch size.
    """
    learning_rate: float = 0.001
    num_iterations: int = 1000
    regularization_strength: float = 0.01
    regularization_type: str = "l2"  # Options could include "l1", "l2", "dropout", etc. for different regularization techniques
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
