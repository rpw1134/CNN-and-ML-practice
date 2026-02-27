from typing import List, Dict, Tuple

from ml_practice_and_neural_net.ml_practice.data_classes.NNHyperparameters import NNHyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNLayerParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNParameters import NNParameters
from ml_practice_and_neural_net.ml_practice.neural_nets.nn_loss import nn_loss_map as loss_map
from ..nn_layers.Layer import Layer

import numpy as np
from numpy.typing import NDArray


class MLP:
    # this class will take in a list of layer configs (or maybe some shorthand, idk)
    # then construct the layer objects, link them together via a list
    # will implement a forward method, backprop method, validate method, and then a train method to combine
    # will take in few hyperparameters: learning rate, epochs, regularization type/strength
    # will use minibatch (can extend later)
    # training should: split into 80/20 train/validate, pass training forward, pass backwards, compute validate error
    # if at any point validation doesn't increase for to many iterations, stop training and return the model with the best validation error (need to track best weights and biases)

    def __init__(self, layer_configs: List[NNLayerParameters], hyperparameters: NNHyperparameters, params: NNParameters):
        self.layers = [Layer(config) for config in layer_configs]
        self.learning_rate = hyperparameters.learning_rate
        self.num_iterations = hyperparameters.num_iterations
        self.regularization_strength = hyperparameters.regularization_strength
        self.regularization_type = hyperparameters.regularization_type
        self.batch_size = hyperparameters.batch_size
        self.loss_function, self.loss_gradient = loss_map[params.loss]

    def feed_forward(self, X: NDArray) -> NDArray:
        output = X
        for layer in self.layers:
            output = layer.feed_forward(output)
        return output

    def predict(self, X: NDArray) -> NDArray:
        output = self.feed_forward(X)
        return output

    def compute_loss(self, predictions: NDArray, Y: NDArray) -> float:
        """Compute the loss given predictions and true labels."""
        return self.loss_function(predictions, Y)

    def compute_gradients(self, predictions: NDArray, Y: NDArray) -> NDArray:
        """Compute the gradient of the loss with respect to the predictions."""

        return self.loss_gradient(predictions, Y)