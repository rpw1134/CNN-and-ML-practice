import numpy as np
from numpy.typing import NDArray

from .BaseModel import BaseModel
from ..data_management.transformations import logistic
from ..loss import ce
from ..data_management.general import split_data, add_data_bias_term
from ..optimization.gradient_descent import gradient_descent


class LogisticRegressionModel(BaseModel):
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

        initial_params = np.random.randn(X.shape[1], 1)

        computed_weights = gradient_descent(init_params=initial_params,
                                            gradient_func=gradient_func,
                                            learning_rate=self.learning_rate,
                                            num_iterations=self.training_iterations)
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
