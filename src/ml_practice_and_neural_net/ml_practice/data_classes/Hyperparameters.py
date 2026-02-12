from dataclasses import dataclass

@dataclass(frozen=True)
class Hyperparameters:
    """
    A data class to encapsulate all hyperparameters required for training a machine learning model.
    This includes the learning rate, number of training iterations, regularization strength, and parameters for optimizers like Adam.
    Unused hyperparameters can be set to their default values, allowing for flexibility across different models and optimization techniques.
    """
    learning_rate: float
    num_training_iterations: int
    lambda_reg: float = 0.0 # Regularization strength (λ), default is 0.0 which means no regularization
    epsilon: float = 1e-8   # Small constant to prevent division by zero in optimizers like Adam
    beta: float = 0.9       # Momentum term for optimizers like Adam
    gamma: float = 0.9      # Decay rate for moving average of squared gradients in optimizers like Adam