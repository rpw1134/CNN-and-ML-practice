from dataclasses import dataclass

from typing_extensions import Literal


@dataclass(frozen=True)
class ModelHyperparameters:
    """
    A data class to encapsulate all hyperparameters required for training a machine learning model.
    This includes the learning rate, number of training iterations, regularization strength, and parameters for optimizers like Adam.
    Unused hyperparameters can be set to their default values, allowing for flexibility across different models and optimization techniques.
    """
    learning_rate: float
    epochs: int
    regularizer: Literal["l1", "l2", "elastic_net"] | None           # The regularization technique to apply during training.
    training_method: Literal["gradient_descent", "minibatch"] | None # The optimization method. Will extend to Adam, RMSProp, etc. in the future.
    lambda_reg: float = 0.0 # Regularization strength (λ), default is 0.0 which means no regularization
    epsilon: float = 1e-8   # Small constant to prevent division by zero in optimizers like Adam
    beta: float = 0.9       # Momentum term for optimizers like Adam
    gamma: float = 0.9      # Decay rate for moving average of squared gradients in optimizers like Adam
    batch_size: int = 32     # Batch size for training, default is 32