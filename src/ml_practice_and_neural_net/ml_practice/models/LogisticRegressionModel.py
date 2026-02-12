import numpy as np
from numpy.typing import NDArray

from .BaseModel import BaseModel
from ..data_management.transformations import logistic
from ..loss import ce
from ..data_management.general import split_data, add_data_bias_term
from ..optimization.gradient_descent import gradient_descent
from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model for binary classification.

    This model uses the logistic (sigmoid) function to predict probabilities for binary outcomes.
    It optimizes parameters using gradient descent with cross-entropy loss.

    Parameters:
        hyperparameters (Hyperparameters): Training hyperparameters such as learning rate and iterations.
        model_data (ModelData | None): Optional metadata including regularization type and training method.

    Attributes:
        weights (NDArray): Learned model parameters (set after calling fit())
        training_set (tuple): Training data and labels
        testing_set (tuple): Testing data and labels

    Example:
        >>> hyperparams = Hyperparameters(learning_rate=0.01, num_training_iterations=1000)
        >>> model_data = ModelData(given_data=X_train, labels=y_train, regularizer="l2", training_method="gradient_descent")
        >>> model = LogisticRegressionModel(hyperparams, model_data)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> train_loss = model.evaluate_ce_training_loss()
    """
    def fit(self, X: NDArray, y: NDArray) -> "LogisticRegressionModel":
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of samples in X and y must be the same. Maybe your labels need to be transposed?")

        X = add_data_bias_term(X)
        training_data, training_labels, testing_data, testing_labels = split_data(X, y)
        self.testing_set = (testing_data, testing_labels)
        self.training_set = (training_data, training_labels)

        _, gradient_func = ce.build(training_data, training_labels)
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

    def evaluate_ce_training_loss(self) -> float:
        return self.evaluate_error(ce.build, use_training_set=True)

    def evaluate_ce_testing_loss(self) -> float:
        return self.evaluate_error(ce.build, use_training_set=False)
