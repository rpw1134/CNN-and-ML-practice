import numpy as np
from numpy.typing import NDArray

from ..data_management.transformations import logistic
from ..loss import ce
from ..data_management.general import split_data, add_data_bias_term
from ..optimization.gradient_descent import gradient_descent
class LogisticRegressionModel:
    def __init__(self, learning_rate: float = 0.01, num_training_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.training_iterations = num_training_iterations
        self.weights = None
        self.training_set = tuple()
        self.testing_set = tuple()
        self.loss_func = None
        self.gradient_func = None

    def fit(self, X: NDArray , y: NDArray ):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of samples in X and y must be the same. Maybe your labels need to be transposed?")

        X = add_data_bias_term(X)
        training_data, training_labels, testing_data, testing_labels = split_data(X, y)
        self.testing_set = (testing_data, testing_labels)
        self.training_set = (training_data, training_labels)

        loss_func, gradient_func = ce.build(training_data, training_labels)
        self.loss_func = loss_func
        self.gradient_func = gradient_func

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
        loss = self.loss_func(self.weights)
        return loss

    def evaluate_ce_testing_loss(self) -> float:
        testing_data, testing_labels = self.testing_set
        loss_func, _ = ce.build(testing_data, testing_labels)
        loss = loss_func(self.weights)
        return loss
