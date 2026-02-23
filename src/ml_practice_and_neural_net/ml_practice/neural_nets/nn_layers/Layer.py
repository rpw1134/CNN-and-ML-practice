from numpy.typing import NDArray
from ml_practice_and_neural_net.ml_practice.data_classes.NNParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.neural_nets.nn_activations import activation_map
from ml_practice_and_neural_net.ml_practice.weight_initializers import initializers_map
import numpy as np


class Layer:
    """Fully connected (dense) neural network layer with optional sparsity masking.

    Forward pass:  z = W @ x + b,  a = activation(z)
        W: (output_dim, input_dim),  x: (input_dim,),  b/z/a: (output_dim,)

    Backward pass (chain rule):
        dL/dW = outer(dL/da * da/dz, x)   shape: (output_dim, input_dim)
        dL/db = dL/da * da/dz              shape: (output_dim,)
        dL/dx = (dL/da * da/dz) @ W        shape: (input_dim,)
    """

    def __init__(self, params: NNLayerParameters):
        # width == output_dim for a dense layer
        self.width = self.output_dim = params.width
        # set by the previous layer width (handled by neural net init)
        self.input_dim = params.input_dim

        self.activation, self.activation_derivative = activation_map[params.activation]()

        # weights: (output_dim, input_dim),  biases: (output_dim,)
        self.weights = initializers_map[params.weight_init](output_dim=self.output_dim, input_dim=self.input_dim)
        self.biases = np.zeros(shape=(self.output_dim,))

        # cached values for backprop
        self.z = None      # pre-activation:  (output_dim,)
        self.a = None      # post-activation: (output_dim,)
        self.input = None  # previous layer output: (input_dim,)

        self.sparsity = params.sparsity
        self.sparsity_mask = (np.random.rand(self.output_dim, self.input_dim) < self.sparsity) if self.sparsity < 1.0 else None

        # needed if output and input dimensions are not equal. projects input onto output so we can add residuals
        self.use_residuals = params.use_residuals
        self.projection_mask = np.ones(shape=(self.output_dim, self.input_dim))

    def feed_forward(self, input_vector: NDArray) -> NDArray:
        """Run one forward pass, caching input/z/a for backprop.

        Args:
            input_vector: (input_dim,)
        Returns:
            a = activation(z): (output_dim,)
        """
        self.input = input_vector
        self.z = (self.weights @ input_vector + self.biases) if not self.sparsity_mask \
            else (self.weights * self.sparsity_mask) @ input_vector + self.biases
        self.a = self.activation(self.z)
        if self.use_residuals:
            self.a += self.projection_mask @ input_vector
        return self.a

    def backprop(self, prev_layer_gradient: NDArray) -> NDArray:
        """Compute gradients, update weights/biases in-place, and return the input gradient.

        Args:
            prev_layer_gradient: dL/da (upstream gradient). shape: (output_dim,)
        Returns:
            dL/dx to pass to the previous layer. shape: (input_dim,)
        """
        delta = prev_layer_gradient * self.activation_derivative(self.z)  # dL/dz, (output_dim,)

        # dL/dW: outer(delta, x) -> (output_dim, input_dim)
        weight_update = np.outer(delta, self.input)
        if self.sparsity_mask is not None:
            weight_update *= self.sparsity_mask

        bias_update = delta  # dL/db, (output_dim,)

        effective_weights = self.weights if self.sparsity_mask is None else self.weights * self.sparsity_mask
        input_gradient = delta @ effective_weights  # dL/dx = delta @ W, shape: (input_dim,)
        if self.use_residuals:
            input_gradient += prev_layer_gradient @ self.projection_mask
            projection_update = np.outer(prev_layer_gradient,self.input)  # dL/d(projection_mask): (output_dim, input_dim)
            self.projection_mask -= projection_update

        self.weights -= weight_update
        self.biases -= bias_update
        return input_gradient
