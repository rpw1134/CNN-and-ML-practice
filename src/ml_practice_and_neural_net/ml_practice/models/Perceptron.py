from ml_practice_and_neural_net.ml_practice.models import BaseModel
import numpy as np
from numpy.typing import NDArray


class Perceptron(BaseModel):

    def fit(self, X: NDArray, y: NDArray):
        raise NotImplementedError("Perceptron model is not implemented yet.")



    def predict(self, X):
        raise NotImplementedError("Perceptron model is not implemented yet.")