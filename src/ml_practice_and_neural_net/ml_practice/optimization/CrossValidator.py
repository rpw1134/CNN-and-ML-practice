from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData
from ..models import model_map
import numpy as np
from numpy.typing import NDArray


class CrossValidator:
    # This will take in a model, a dataset, a set of hyperparameters, a regularization strategy, an optimization strategy, and a number of folds k
    # it will split the dataset into k
    # it will take all possible k-1, 1 splits and train the given model on the k-1 and test it on the 1
    # will store these results and return the average of the k results
    def __init__(self, model: ModelData, hyperparams: Hyperparameters, k: int):
        self.model = model_map[model.model_name](hyperparams)
        self.hyperparams = hyperparams
        self.data: NDArray = model.given_data
        self.labels: NDArray = model.labels
        self.k = k
        self.errors = []
        self.mean_error = None
        self.accuracies = []
        self.mean_accuracy = None
        self.std_dev = None

    def cross_validate(self):
        """
        Perform k-fold cross-validation on the given model and dataset.

        :return: The instance, now with populated error and accuracy metrics for each fold.
        """
        n_samples = self.data.shape[0]
        fold_size = n_samples // self.k
        # indices for each fold
        indices = [[i * fold_size, (i + 1) * fold_size] for i in range(self.k)]
        # add remainder to the last fold
        indices[-1][1] += n_samples % self.k
        for i in range(self.k):
            # split train and test
            test_samples = self.data[indices[i][0]:indices[i][1]]
            test_labels = self.labels[indices[i][0]:indices[i][1]]
            train_samples = self.data[indices[i][1]:] if i == 0 else np.concatenate((self.data[:indices[i][0]], self.data[indices[i][1]:]), axis=0)
            train_labels = self.labels[indices[i][1]:] if i == 0 else np.concatenate((self.labels[:indices[i][0]], self.labels[indices[i][1]:]), axis=0)

            self.model.fit(train_samples, train_labels)
            predictions = self.model.predict(test_samples)
            error = np.mean((predictions - test_labels) ** 2)
            accuracy = np.mean(predictions == test_labels)
            self.errors.append(error)
            self.accuracies.append(accuracy)
        return self


    def get_metrics_mean_metrics(self):
        """
        Calculate and return the mean error, mean accuracy, and standard deviation of errors across all folds, or return them if previously calculated.
        :return: A dictionary containing the mean error, mean accuracy, and standard deviation of errors across all folds.
        """
        if self.mean_error is None:
            self.mean_error = sum(self.errors) / len(self.errors)
        if self.mean_accuracy is None:
            self.mean_accuracy = sum(self.accuracies) / len(self.accuracies)
        if self.std_dev is None:
            self.std_dev = (sum((x - self.mean_error) ** 2 for x in self.errors) / len(self.errors)) ** 0.5
        return {
            "mean_error": self.mean_error,
            "mean_accuracy": self.mean_accuracy,
            "std_dev": self.std_dev
        }

    def get_errors_and_accuracies(self):
        """
        Return the list of errors and accuracies for each fold.
        :return: A dictionary containing the list of errors and accuracies for each fold.
        """
        return {
            "errors": self.errors,
            "accuracies": self.accuracies
        }
