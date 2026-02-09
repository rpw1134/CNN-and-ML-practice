from ml_practice_and_neural_net.ml_practice.data_management.general import add_data_bias_term, split_data
import numpy as np

class SoftmaxRegressionModel:
    def __init__(self, learning_rate: float = 0.01, num_training_iterations: int = 1000):
        self.categories = None
        self.learning_rate = learning_rate
        self.training_iterations = num_training_iterations
        self.weights = None
        self.training_set = tuple()
        self.testing_set = tuple()
        self.loss_func = None
        self.gradient_func = None

    def fit(self, X, y):
        if X.ndim != 2:
            np.reshape(X, (X.shape[0], 1))
        X = add_data_bias_term(X)

        categories = np.unique(y)
        num_categories = len(categories)

        # map to index position in one hot
        self.categories = {cat: ind for ind, cat in enumerate(categories)}

        training_data, training_labels, testing_data, testing_labels = split_data(X, y)
        self.testing_set = (testing_data, testing_labels)
        self.training_set = (training_data, training_labels)
