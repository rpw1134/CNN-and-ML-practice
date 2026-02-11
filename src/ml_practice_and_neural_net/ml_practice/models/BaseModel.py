from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from numpy.typing import NDArray

from ml_practice_and_neural_net.ml_practice.regularization import regularization_map


class BaseModel(ABC):
    """
    Abstract base class for machine learning models.

    Subclasses must implement fit() and predict() methods.
    The evaluate_error() method is provided as a concrete implementation.
    """

    def __init__(self, learning_rate: float = 0.01, num_training_iterations: int = 1000,
                 reg_technique: str | None = None, lambda_reg: float = 0.01) -> None:
        """
        Initialize the base model with training hyperparameters.

        Args:
            learning_rate: Step size for gradient descent optimization.
                          Higher values may speed up convergence but risk overshooting.
                          Default: 0.01
            num_training_iterations: Maximum number of gradient descent iterations to perform.
                                    Training may stop earlier if convergence is detected.
                                    Default: 1000
            reg_technique: Type of regularization to apply. Options:
                          - "l1": L1 regularization (Lasso) - encourages sparsity, zeros out small weights
                          - "l2": L2 regularization (Ridge) - penalizes large weights, prevents overfitting
                          - "elastic_net": Combination of L1 and L2 regularization
                          - None: No regularization applied
                          Default: None
            lambda_reg: Regularization strength (λ). Higher values increase the penalty on weights.
                       Only used if reg_technique is specified.
                       Default: 0.01
        """
        self.learning_rate = learning_rate
        self.training_iterations = num_training_iterations
        self.weights: Optional[NDArray] = None
        self.training_set: Tuple = ()
        self.testing_set: Tuple = ()
        self.reg_technique = regularization_map(reg_technique) if reg_technique else None
        self.lambda_reg = lambda_reg


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
        regularization_loss, _ = self.reg_technique(self.lambda_reg) if self.reg_technique else (lambda x: 0, lambda x: 0)
        loss = loss_func(self.weights) + regularization_loss(self.weights)
        return loss

