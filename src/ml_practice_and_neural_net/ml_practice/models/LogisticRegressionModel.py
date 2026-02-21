import numpy as np
from numpy.typing import NDArray

from .BaseModel import BaseModel
from ..data_management.transformations import logistic
from ..loss import ce
from ..data_management.general import add_data_bias_term
from ..optimization.gradient_descent import gradient_descent
from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model for binary classification.

    This model uses the logistic (sigmoid) function to predict probabilities for binary outcomes.
    It optimizes parameters using gradient descent with cross-entropy loss.

    Parameters:
        hyperparameters (Hyperparameters): Training hyperparameters including learning rate, epochs,
                                           regularization strategy, and training method.
        model_data (ModelData | None): Optional metadata (currently unused, kept for backward compatibility).

    Attributes:
        weights (NDArray): Learned model parameters (set after calling fit())
        training_set (tuple): Training data and labels
    Example:
        >>> hyperparams = Hyperparameters(learning_rate=0.01, epochs=1000, regularizer="l2", training_method="gradient_descent")
        >>> model = LogisticRegressionModel(hyperparams)
        >>> model.fit(X, y)
        >>> predictions = model.predict(X_test)
        >>> train_loss = model.training_ce()
        >>> eval_loss = model.evaluate_ce(X_test, y_test)
    """
    def fit(self, X: NDArray, y: NDArray) -> "LogisticRegressionModel":
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of samples in X and y must be the same. Maybe your labels need to be transposed?")

        self.training_set = (X, y)
        X = add_data_bias_term(X)

        _, gradient_func = ce.build(X, y)
        _, regularization_gradient = self.reg_technique(self.lambda_reg) if self.reg_technique else (None, lambda x: np.zeros_like(x))

        initial_params = np.random.randn(X.shape[1], 1)

        computed_weights = gradient_descent(init_params=initial_params,
                                            gradient_func=gradient_func,
                                            learning_rate=self.learning_rate,
                                            num_iterations=self.training_iterations,
                                            regularization_gradient=regularization_gradient)
        self.weights = computed_weights
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = add_data_bias_term(X)
        logits = X @ self.weights
        return logistic(logits)

    def evaluate_ce(self, X: NDArray, y: NDArray) -> float:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        probs = self.predict(X)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))

    def training_ce(self) -> float:
        return self.evaluate_ce(*self.training_set)
