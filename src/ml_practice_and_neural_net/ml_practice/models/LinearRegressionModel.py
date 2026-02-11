import numpy as np
from numpy.typing import NDArray

from .BaseModel import BaseModel
from ..loss import mse
from ..data_management.general import split_data, add_data_bias_term
from ..optimization.gradient_descent import gradient_descent


class LinearRegressionModel(BaseModel):
    """
    Linear Regression model for continuous value prediction.

    This model predicts continuous outputs using a linear function of the input features.
    It optimizes parameters using gradient descent with mean squared error loss.

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

    Example:
        >>> model = LinearRegressionModel(learning_rate=0.01,
        ...                                num_training_iterations=1000,
        ...                                reg_technique="l2",
        ...                                lambda_reg=0.1)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> train_loss = model.evaluate_mse_training_loss()
    """
    def fit(self, X: NDArray, y: NDArray) -> "LinearRegressionModel":
        """
        Train the linear regression model on the given data.

        Args:
            X: Input features, shape (n_samples, n_features) or (n_samples,)
            y: Target values, shape (n_samples, 1) or (n_samples,)

        Returns:
            self: Returns the instance itself for method chaining
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of samples in X and y must be the same. Maybe your labels need to be transposed?")

        X = add_data_bias_term(X)
        training_data, training_labels, testing_data, testing_labels = split_data(X, y)
        self.testing_set = (testing_data, testing_labels)
        self.training_set = (training_data, training_labels)

        _, gradient_func = mse.build(training_data, training_labels)
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
        """
        Make predictions on the given data.

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predictions, shape (n_samples, 1)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = add_data_bias_term(X)
        predictions = X @ self.weights
        return predictions

    def evaluate_mse_training_loss(self) -> float:
        """
        Evaluate the mean squared error on the training set.

        Returns:
            MSE loss value including regularization penalty if applicable
        """
        return self.evaluate_error(mse.build, use_training_set=True)

    def evaluate_mse_testing_loss(self) -> float:
        """
        Evaluate the mean squared error on the testing set.

        Returns:
            MSE loss value including regularization penalty if applicable
        """
        return self.evaluate_error(mse.build, use_training_set=False)

    def calculate_r2_score(self, X: NDArray, y: NDArray) -> float:
        """
        Calculate the R² (coefficient of determination) score.

        R² = 1 - (SS_residual / SS_total)
        where SS_residual = sum((y - y_pred)²) and SS_total = sum((y - y_mean)²)

        Args:
            X: Input features
            y: True target values

        Returns:
            R² score (1.0 is perfect fit, 0.0 means no better than mean baseline)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        y_pred = self.predict(X)
        ss_residual = np.sum((y - y_pred) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_residual / ss_total)
        return float(r2)

