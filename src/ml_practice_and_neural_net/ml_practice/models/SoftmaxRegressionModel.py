from ml_practice_and_neural_net.ml_practice.data_management.encoding import convert_to_one_hot
from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term, split_data
import numpy as np
from ..loss import cce
from ..optimization.gradient_descent import gradient_descent
from numpy.typing import NDArray


class SoftmaxRegressionModel:
    def __init__(self, learning_rate: float = 0.01, num_training_iterations: int = 1000):
        self.index_to_categories = None
        self.learning_rate = learning_rate
        self.training_iterations = num_training_iterations
        self.weights = None
        self.training_set = tuple()
        self.testing_set = tuple()

    def fit(self, X, y):
        if X.ndim != 2:
            np.reshape(X, (X.shape[0], 1))

        # data transformations
        X = add_data_bias_term(X)
        one_hot_labels = convert_to_one_hot(y)

        # Look up for predictions
        categories = np.unique(y)
        self.index_to_categories = {i: cat for i, cat in enumerate(categories)}

        # sets in terms of one hot labels
        training_data, training_labels, testing_data, testing_labels = split_data(X, one_hot_labels)
        self.testing_set = (testing_data, testing_labels)
        self.training_set = (training_data, training_labels)

        loss, gradient = cce.build(training_data, training_labels)

        # initialize parameters with small random values, shape should be CxD where C is number of categories and D is number of features (including bias)
        initial_params = np.random.randn(one_hot_labels.shape[1], X.shape[1])

        self.weights = gradient_descent(init_params=initial_params,
                                        learning_rate=self.learning_rate,
                                        gradient_func=gradient,
                                        num_iterations=self.training_iterations)
        return self

    def predict(self, X: NDArray):
        X = add_data_bias_term(X)
        logits = X @ self.weights.T
        index = np.argmax(logits, axis=1)
        return self.index_to_categories[index]

    def evaluate_cce_training_loss(self) -> float:
        training_data, training_labels = self.training_set
        loss_func, _ = cce.build(training_data, training_labels)
        loss = loss_func(self.weights)
        return loss

    def evaluate_cce_testing_loss(self) -> float:
        testing_data, testing_labels = self.testing_set
        loss_func, _ = cce.build(testing_data, testing_labels)
        loss = loss_func(self.weights)
        return loss



