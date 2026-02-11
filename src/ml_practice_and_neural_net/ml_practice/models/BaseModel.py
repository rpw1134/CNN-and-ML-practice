from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from numpy.typing import NDArray


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    Subclasses must implement fit() and predict() methods.
    The evaluate_error() method is provided as a concrete implementation.
    """

    def __init__(self, learning_rate: float = 0.01, num_training_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.training_iterations = num_training_iterations
        self.weights: Optional[NDArray] = None
        self.training_set: Tuple = ()
        self.testing_set: Tuple = ()

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

    def evaluate_error(
        self,
        loss_builder: Callable[[NDArray, NDArray], Tuple[Callable[[NDArray], float], Callable[[NDArray], NDArray]]],
        use_training_set: bool = True
    ) -> float:
        """
        Evaluate the error on either the training or testing set.

        Args:
            loss_builder: A function that takes (X, y) and returns (loss_func, gradient_func)
            use_training_set: If True, evaluate on training set; if False, evaluate on testing set

        Returns:
            The computed loss value

        Raises:
            ValueError: If the model hasn't been fitted yet or the requested dataset is empty
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before evaluating error")

        dataset = self.training_set if use_training_set else self.testing_set

        if not dataset or len(dataset) != 2:
            raise ValueError(f"{'Training' if use_training_set else 'Testing'} set is empty or invalid")

        data, labels = dataset
        loss_func, _ = loss_builder(data, labels)
        loss = loss_func(self.weights)
        return loss

