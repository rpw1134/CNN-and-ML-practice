from ml_practice_and_neural_net.ml_practice.data_classes.Hyperparameters import Hyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.ModelData import ModelData
from ..models import model_map


class CrossValidator:
    # This will take in a model, a dataset, a set of hyperparameters, a regularization strategy, an optimization strategy, and a number of folds k
    # it will split the dataset into k
    # it will take all possible k-1, 1 splits and train the given model on the k-1 and test it on the 1
    # will store these results and return the average of the k results
    def __init__(self, model: ModelData, hyperparams: Hyperparameters, k: int):
        self.model = model_map[model.model_name](hyperparams)
        self.hyperparams = hyperparams
        self.data = model.given_data
        self.labels = model.labels
        self.k = k
        self.errors = []
        self.accuracies = []
        self.std_devs = []

