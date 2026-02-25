from abc import ABC, abstractmethod
from typing import Tuple, Optional
from numpy.typing import NDArray

from ml_practice_and_neural_net.ml_practice.data_classes.ModelHyperparameters import ModelHyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData
from ml_practice_and_neural_net.ml_practice.regularization import regularization_map


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    Subclasses must implement fit() and predict() methods.
    """

    def __init__(self, hyperparameters: ModelHyperparameters) -> None:
        """
        Initialize the base model with dataclass-based training configuration.

        Args:
            hyperparameters: Training hyperparameters (learning rate, epochs, regularization strategy, training method, etc.).
        """
        self.hyperparameters = hyperparameters
        self.learning_rate = hyperparameters.learning_rate
        self.training_iterations = hyperparameters.epochs
        self.weights: Optional[NDArray] = None
        self.training_set: Tuple = ()

        reg_choice = hyperparameters.regularizer
        self.reg_technique = regularization_map[reg_choice] if reg_choice else None
        self.lambda_reg = hyperparameters.lambda_reg
        self.training_method = hyperparameters.training_method

    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> "BaseModel":
        """
        Train the model on the given data.

        Args:
            X: Input features, shape (n_samples, n_features)
            y: Target values, shape (n_samples,) or (n_samples, n_outputs)

        Returns:
            self: Returns the instance itself for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """
        Make predictions on the given data.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape depends on the specific model
        """
        pass


