from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term
from ml_practice_and_neural_net.ml_practice.models import BaseModel
import numpy as np
from numpy.typing import NDArray


class Perceptron(BaseModel):
    # only uses epochs and learning rate, no regularization or training method
    def fit(self, X: NDArray, y: NDArray):
        if X.ndim != 2:
            X = X.reshape(-1, 1)
        X = add_data_bias_term(X)
        if y.ndim != 2:
            y = y.reshape(-1, 1)
        self.weights = np.random.uniform(low=-1, high=1, size=(X.shape[1], 1))
        for epoch in range(self.training_iterations):
            predictions = np.sign(X @ self.weights)
            misclassified = (predictions != y).flatten()
            if not np.any(misclassified):
                break
            self.weights += self.learning_rate * (X[misclassified].T @ y[misclassified])


    def predict(self, X):
        return np.sign(X @ self.weights)