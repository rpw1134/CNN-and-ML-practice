from ..models import BaseModel
import numpy as np
from numpy.typing import NDArray


class CrossValidator:
    """
    Performs k-fold cross-validation on a given model and dataset.

    Splits the dataset into k folds, trains on k-1 folds and tests on the remaining fold.
    Tracks both training and testing performance metrics across all folds.
    """

    def __init__(self, model: BaseModel, data: NDArray, labels: NDArray, k: int):
        """
        Initialize the CrossValidator.

        :param model: An instance of a BaseModel subclass to evaluate.
        :param data: Input features as a numpy array.
        :param labels: Target labels as a numpy array.
        :param k: Number of folds for cross-validation.
        """
        self.model = model
        self.data: NDArray = data
        self.labels: NDArray = labels
        self.k = k

        # Training metrics
        self.train_errors = []
        self.train_accuracies = []
        self.mean_train_error = None
        self.mean_train_accuracy = None
        self.train_std_dev = None

        # Testing metrics
        self.test_errors = []
        self.test_accuracies = []
        self.mean_test_error = None
        self.mean_test_accuracy = None
        self.test_std_dev = None

    def cross_validate(self):
        """
        Perform k-fold cross-validation on the given model and dataset.

        For each fold, trains the model on k-1 folds and evaluates on both the training
        set and the held-out test fold. Collects error and accuracy metrics for both.

        :return: The instance, now with populated training and testing metrics for each fold.
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

            # Calculate training metrics
            train_predictions = self.model.predict(train_samples)
            train_error = np.mean((train_predictions - train_labels) ** 2)
            train_accuracy = np.mean(train_predictions == train_labels)
            self.train_errors.append(train_error)
            self.train_accuracies.append(train_accuracy)

            # Calculate testing metrics
            test_predictions = self.model.predict(test_samples)
            test_error = np.mean((test_predictions - test_labels) ** 2)
            test_accuracy = np.mean(test_predictions == test_labels)
            self.test_errors.append(test_error)
            self.test_accuracies.append(test_accuracy)
        return self


    def get_train_metrics(self):
        """
        Calculate and return the mean training error, accuracy, and standard deviation across all folds.

        :return: A dictionary containing the mean training error, mean training accuracy,
                 and standard deviation of training errors across all folds.
        """
        if self.mean_train_error is None:
            self.mean_train_error = sum(self.train_errors) / len(self.train_errors)
        if self.mean_train_accuracy is None:
            self.mean_train_accuracy = sum(self.train_accuracies) / len(self.train_accuracies)
        if self.train_std_dev is None:
            self.train_std_dev = (sum((x - self.mean_train_error) ** 2 for x in self.train_errors) / len(self.train_errors)) ** 0.5
        return {
            "mean_error": self.mean_train_error,
            "mean_accuracy": self.mean_train_accuracy,
            "std_dev": self.train_std_dev
        }

    def get_test_metrics(self):
        """
        Calculate and return the mean testing error, accuracy, and standard deviation across all folds.

        :return: A dictionary containing the mean testing error, mean testing accuracy,
                 and standard deviation of testing errors across all folds.
        """
        if self.mean_test_error is None:
            self.mean_test_error = sum(self.test_errors) / len(self.test_errors)
        if self.mean_test_accuracy is None:
            self.mean_test_accuracy = sum(self.test_accuracies) / len(self.test_accuracies)
        if self.test_std_dev is None:
            self.test_std_dev = (sum((x - self.mean_test_error) ** 2 for x in self.test_errors) / len(self.test_errors)) ** 0.5
        return {
            "mean_error": self.mean_test_error,
            "mean_accuracy": self.mean_test_accuracy,
            "std_dev": self.test_std_dev
        }

    def get_all_metrics(self):
        """
        Calculate and return both training and testing metrics.

        :return: A dictionary with 'train' and 'test' keys, each containing mean error,
                 mean accuracy, and standard deviation.
        """
        return {
            "train": self.get_train_metrics(),
            "test": self.get_test_metrics()
        }

    def get_errors_and_accuracies(self):
        """
        Return the list of errors and accuracies for each fold, for both training and testing.

        :return: A dictionary containing lists of training and testing errors and accuracies for each fold.
        """
        return {
            "train_errors": self.train_errors,
            "train_accuracies": self.train_accuracies,
            "test_errors": self.test_errors,
            "test_accuracies": self.test_accuracies
        }
