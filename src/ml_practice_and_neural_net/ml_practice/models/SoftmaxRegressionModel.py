from ml_practice_and_neural_net.ml_practice.data_management.encoding import convert_to_one_hot
from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term, split_data
import numpy as np
from .BaseModel import BaseModel
from ..loss import cce
from ..optimization.gradient_descent import gradient_descent
from numpy.typing import NDArray
from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData


class SoftmaxRegressionModel(BaseModel):
    """
    Softmax Regression model for multi-class classification.

    This model uses the softmax function to predict probabilities across multiple classes.
    It optimizes parameters using gradient descent with categorical cross-entropy loss.

    Parameters:
        hyperparameters (Hyperparameters): Training hyperparameters such as learning rate and iterations.
        model_data (ModelData | None): Optional metadata including regularization type and training method.

    Attributes:
        weights (NDArray): Learned model parameters (set after calling fit())
        training_set (tuple): Training data and labels
        testing_set (tuple): Testing data and labels
        index_to_categories (dict): Mapping from class indices to original category labels

    Example:
        >>> hyperparams = Hyperparameters(learning_rate=0.01, num_training_iterations=1000)
        >>> model_data = ModelData(given_data=X_train, labels=y_train, regularizer="l2", training_method="gradient_descent")
        >>> model = SoftmaxRegressionModel(hyperparams, model_data)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)  # Returns original category labels
        >>> train_loss = model.evaluate_cce_training_loss()
    """
    def __init__(self, hyperparameters: Hyperparameters, model_data: ModelData | None = None):
        super().__init__(hyperparameters, model_data)
        self.index_to_categories = None

    def fit(self, X, y) -> "SoftmaxRegressionModel":
        if X.ndim != 2:
            X = X.reshape(X.shape[0], 1)

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

        _, gradient = cce.build(training_data, training_labels)
        _, regularization_gradient = self.reg_technique(self.lambda_reg) if self.reg_technique else (None, lambda x: np.zeros_like(x))

        # initialize parameters with small random values, shape should be CxD where C is number of categories and D is number of features (including bias)
        initial_params = 0.01 * np.random.randn(one_hot_labels.shape[1], X.shape[1])

        self.weights = gradient_descent(init_params=initial_params,
                                        learning_rate=self.learning_rate,
                                        gradient_func=gradient,
                                        num_iterations=self.training_iterations,
                                        regularization_gradient=regularization_gradient)
        return self

    def predict(self, X: NDArray) -> NDArray:
        X = add_data_bias_term(X)
        logits = X @ self.weights.T
        index = np.argmax(logits, axis=1)
        return np.array([self.index_to_categories[i] for i in index])

    def evaluate_cce_training_loss(self) -> float:
        return self.evaluate_error(cce.build, use_training_set=True)

    def evaluate_cce_testing_loss(self) -> float:
        return self.evaluate_error(cce.build, use_training_set=False)

