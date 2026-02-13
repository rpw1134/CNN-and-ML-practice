from typing import List, Dict

import numpy as np

from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData
from ml_practice_and_neural_net.ml_practice.models import model_map
from .CrossValidator import CrossValidator
from ..data_management.general import shuffle_data


class HyperparameterTuner:
    """
    Performs hyperparameter tuning using k-fold cross-validation.

    This class evaluates multiple hyperparameter configurations by training models
    with different parameter sets and comparing their cross-validation performance.
    The best hyperparameter set is selected based on the lowest mean error.
    """

    def __init__(self, model_data: ModelData, parameter_grid: List[Dict], k:int) -> None:
        """
        Initialize the HyperparameterTuner.

        :param model_data: ModelData object containing the model name, training data, and labels.
        :param parameter_grid: List of dictionaries, where each dictionary contains a set of
                               hyperparameters to evaluate (e.g., [{'learning_rate': 0.01, 'epochs': 100}, ...]).
        :param k: Number of folds to use for cross-validation.
        """
        self.model = model_map[model_data.model_name]
        self.model_data = model_data.given_data
        self.model_labels = model_data.labels
        self.params_grid = parameter_grid
        self.k = k
        self.mean_stats = []
        self.errors_and_accuracies = []
        self.best_params = None

    def execute_parameter_search(self):
        """
        Execute hyperparameter search using k-fold cross-validation.

        For each set of hyperparameters in the parameter grid, this method:
        1. Creates a new model instance with those hyperparameters
        2. Performs k-fold cross-validation on the shuffled dataset
        3. Collects mean error, mean accuracy, and standard deviation metrics
        4. Identifies the best hyperparameter set based on lowest mean error

        The data is shuffled once before evaluation to ensure fair comparison across
        all parameter sets.

        :return: Self, with populated mean_stats, errors_and_accuracies, and best_params attributes.
        """
        # Shuffle data once before cross-validation to ensure fair comparison across all parameter sets
        shuffled_data, shuffled_labels = shuffle_data(self.model_data, self.model_labels)

        for param_set in self.params_grid:
            new_model = self.model(**param_set)
            cross_validator = CrossValidator(new_model, shuffled_data, shuffled_labels, self.k).cross_validate()
            self.mean_stats.append(cross_validator.get_mean_metrics())
            self.errors_and_accuracies.append(cross_validator.get_errors_and_accuracies())

        # Find best parameters based on lowest mean error
        mean_errors = [stats["mean_error"] for stats in self.mean_stats]
        best_model_index = np.argmin(mean_errors)
        self.best_params = self.params_grid[best_model_index]
        return self

    def get_diagnostics(self):
        """
        Get detailed diagnostic statistics for all evaluated hyperparameter sets.

        Returns a dictionary where keys are formatted strings representing each hyperparameter
        configuration (e.g., "learning_rate=0.01, epochs=100"), and values are dictionaries
        containing the mean error, mean accuracy, and standard deviation of errors across all folds.

        :return: Dictionary mapping hyperparameter configurations to their performance metrics.
                 Each value contains: {"mean_error": float, "mean_accuracy": float, "std_dev": float}
        """
        diagnostics = dict()
        for i, param_set in enumerate(self.params_grid):
            formatted_str = ", ".join(f"{key}={value}" for key, value in param_set.items())
            diagnostics[formatted_str] = {"mean_error": self.mean_stats[i]["mean_error"], "mean_accuracy": self.mean_stats[i]["mean_accuracy"], "std_dev": self.mean_stats[i]["std_dev"]}
        return diagnostics