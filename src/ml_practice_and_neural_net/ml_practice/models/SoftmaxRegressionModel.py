from ml_practice_and_neural_net.ml_practice.data_management.encoding import convert_to_one_hot
from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term, split_data
import numpy as np
from .BaseModel import BaseModel
from ..loss import cce
from ..optimization.gradient_descent import gradient_descent
from numpy.typing import NDArray


class SoftmaxRegressionModel(BaseModel):
    """
    Softmax Regression model for multi-class classification.

    This model uses the softmax function to predict probabilities across multiple classes.
    It optimizes parameters using gradient descent with categorical cross-entropy loss.

    Parameters:
        learning_rate (float): Step size for gradient descent. Default: 0.01
        num_training_iterations (int): Maximum number of training iterations. Default: 1000
        reg_technique (str | None): Regularization type. Options:
            - "l1": L1 regularization (Lasso) - encourages sparse weights
            - "l2": L2 regularization (Ridge) - prevents large weights
            - "elastic_net": Combination of L1 and L2
            - None: No regularization
            Default: None
        lambda_reg (float): Regularization strength (λ). Default: 0.01

    Attributes:
        weights (NDArray): Learned model parameters (set after calling fit())
        training_set (tuple): Training data and labels
        testing_set (tuple): Testing data and labels
        index_to_categories (dict): Mapping from class indices to original category labels

    Example:
        >>> model = SoftmaxRegressionModel(learning_rate=0.01,
        ...                                 num_training_iterations=1000,
        ...                                 reg_technique="l2",
        ...                                 lambda_reg=0.1)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)  # Returns original category labels
        >>> train_loss = model.evaluate_cce_training_loss()
    """
    def __init__(self, learning_rate: float = 0.01, num_training_iterations: int = 1000,
                 reg_technique: str | None = None, lambda_reg: float = 0.01):
        super().__init__(learning_rate, num_training_iterations, reg_technique, lambda_reg)
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



