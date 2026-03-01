from numpy.typing import NDArray
from ml_practice_and_neural_net.ml_practice.data_classes.NNLayerParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.neural_nets.nn_activations import activation_map
from ml_practice_and_neural_net.ml_practice.weight_initializers import initializers_map
import numpy as np


class Layer:
    """Fully connected (dense) neural network layer with optional sparsity masking.

    Supports both single samples and mini-batches. A 1D input (input_dim,) is
    automatically treated as a batch of size 1.

    Forward pass:  Z = X @ W.T + b,  A = activation(Z)
        W: (output_dim, input_dim)
        X: (batch_size, input_dim)
        b/Z/A: (batch_size, output_dim)

    Backward pass (chain rule, averaged over batch):
        dL/dW = delta.T @ X / batch_size   shape: (output_dim, input_dim)
        dL/db = mean(delta, axis=0)        shape: (output_dim,)
        dL/dX = delta @ W                  shape: (batch_size, input_dim)

    Loss gradient functions return raw (un-averaged) gradients; averaging over
    the batch is performed here before applying the learning rate.

    Residual connection:
        When input_dim == output_dim the skip is a fixed identity — no parameters.
        When input_dim != output_dim a learned projection matrix W_proj
        (xavier-initialised, shape (output_dim, input_dim)) is used and trained
        alongside the main weights.
    """

    def __init__(self, params: NNLayerParameters, learning_rate: float = 0.001):
        self.width = self.output_dim = params.width
        self.input_dim = params.input_dim
        self.learning_rate = learning_rate

        self.activation, self.activation_derivative = activation_map[params.activation]()

        self.weights = initializers_map[params.weight_init](output_dim=self.output_dim, input_dim=self.input_dim)
        self.biases = np.zeros(shape=(self.output_dim,))

        # cached values for backprop — all (batch_size, dim)
        self.z = None
        self.a = None
        self.input = None

        self.sparsity = params.sparsity
        self.sparsity_mask = (np.random.rand(self.output_dim, self.input_dim) < self.sparsity) if self.sparsity < 1.0 else None

        self.use_residuals = params.use_residuals
        if self.use_residuals:
            if self.input_dim == self.output_dim:
                # fixed identity — no parameters to learn
                self.projection_weights = None
            else:
                # learned linear projection to match dimensions
                self.projection_weights = initializers_map["xavier_normal"](
                    output_dim=self.output_dim, input_dim=self.input_dim
                )

    def feed_forward(self, input_vector: NDArray) -> NDArray:
        """Run one forward pass, caching input/z/a for backprop.

        Args:
            input_vector: (input_dim,) or (batch_size, input_dim)
        Returns:
            a = activation(z): (batch_size, output_dim)
        """
        # normalise to 2D: (batch_size, input_dim)
        if input_vector.ndim == 1:
            input_vector = input_vector[np.newaxis, :]  # (1, input_dim)

        self.input = input_vector  # (batch_size, input_dim)

        effective_weights = self.weights if self.sparsity_mask is None else self.weights * self.sparsity_mask
        # X @ W.T -> (batch_size, output_dim)
        self.z = self.input @ effective_weights.T + self.biases
        self.a = self.activation(self.z)

        if self.use_residuals:
            # identity shortcut when dims match, else learned projection
            skip = self.input if self.projection_weights is None else self.input @ self.projection_weights.T
            self.a += skip  # (batch_size, output_dim)

        return self.a  # (batch_size, output_dim)

    def backprop(self, prev_layer_gradient: NDArray) -> NDArray:
        """Compute gradients, update weights/biases in-place, and return the input gradient.

        Args:
            prev_layer_gradient: dL/dA, shape: (batch_size, output_dim)
        Returns:
            dL/dX to pass to the previous layer, shape: (batch_size, input_dim)
        """
        # normalise to 2D in case a single sample gradient was passed
        if prev_layer_gradient.ndim == 1:
            prev_layer_gradient = prev_layer_gradient[np.newaxis, :]  # (1, output_dim)

        batch_size = self.input.shape[0]

        delta = prev_layer_gradient * self.activation_derivative(self.z)  # (batch_size, output_dim)

        # dL/dW averaged over batch: delta.T @ X / batch_size -> (output_dim, input_dim)
        weight_update = delta.T @ self.input / batch_size
        if self.sparsity_mask is not None:
            weight_update *= self.sparsity_mask

        bias_update = delta.mean(axis=0)  # (output_dim,)

        effective_weights = self.weights if self.sparsity_mask is None else self.weights * self.sparsity_mask
        input_gradient = delta @ effective_weights  # (batch_size, input_dim)

        if self.use_residuals:
            if self.projection_weights is None:
                # identity shortcut — gradient passes straight through, no parameters
                input_gradient += prev_layer_gradient
            else:
                # learned projection — propagate gradient and update weights
                input_gradient += prev_layer_gradient @ self.projection_weights  # (batch_size, input_dim)
                proj_update = prev_layer_gradient.T @ self.input / batch_size    # (output_dim, input_dim)
                self.projection_weights -= self.learning_rate * proj_update

        self.weights -= self.learning_rate * weight_update
        self.biases -= self.learning_rate * bias_update

        return input_gradient  # (batch_size, input_dim)
