from typing import List, Dict, Tuple

from ml_practice_and_neural_net.ml_practice.data_classes.NNHyperparameters import NNHyperparameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNLayerParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.data_classes.NNParameters import NNParameters
from ml_practice_and_neural_net.ml_practice.neural_nets.nn_loss import nn_loss_map as loss_map
from ..nn_layers.Layer import Layer
from numpy.typing import NDArray

from ...data_management.general import split_data, shuffle_data


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
        self.early_stopping_patience = hyperparameters.early_stopping_patience
        self.epochs = hyperparameters.epochs

    def feed_forward(self, X: NDArray) -> NDArray:
        """Run a forward pass through the network."""
        output = X
        for layer in self.layers:
            output = layer.feed_forward(output)
        return output

    def predict(self, X: NDArray) -> NDArray:
        """Run a forward pass through the network to get predictions."""
        output = self.feed_forward(X)
        return output

    def compute_loss(self, predictions: NDArray, Y: NDArray) -> float:
        """Compute the loss given predictions and true labels."""
        return self.loss_function(predictions, Y)

    def compute_gradients(self, predictions: NDArray, Y: NDArray) -> NDArray:
        """Compute the gradient of the loss with respect to the predictions."""
        return self.loss_gradient(predictions, Y)

    def _backprop(self, predictions: NDArray, Y: NDArray) -> None:
        """Compute loss gradient and propagate it backward through all layers."""
        grad = self.compute_gradients(predictions, Y)
        for layer in reversed(self.layers):
            grad = layer.backprop(grad)

    def train(self, X: NDArray, Y: NDArray) -> None:
        """Train the MLP using mini-batch gradient descent with early stopping."""

        # reserve validation set once before training
        train_data, train_labels, val_data, val_labels = split_data(data=X, labels=Y, shuffle=True, test_ratio=0.2)

        best_val_loss = float('inf')
        best_weights = [layer.weights.copy() for layer in self.layers]
        best_biases = [layer.biases.copy() for layer in self.layers]
        patience_counter = 0

        for epoch in range(self.epochs):
            # shuffle training data each epoch
            train_data, train_labels = shuffle_data(train_data, train_labels)

            # mini-batch loop
            n = train_data.shape[0]
            for start in range(0, n, self.batch_size):
                X_batch = train_data[start:start + self.batch_size]
                Y_batch = train_labels[start:start + self.batch_size]

                # forward pass
                predictions = self.feed_forward(X_batch)

                self._backprop(predictions, Y_batch)

            # validation pass — no backward
            val_predictions = self.predict(val_data)
            val_loss = self.compute_loss(val_predictions, val_labels)

            # early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = [layer.weights.copy() for layer in self.layers]
                best_biases = [layer.biases.copy() for layer in self.layers]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break

        # restore best weights
        for layer, w, b in zip(self.layers, best_weights, best_biases):
            layer.weights = w
            layer.biases = b

