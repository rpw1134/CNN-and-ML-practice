from numpy.typing import NDArray
from ml_practice_and_neural_net.ml_practice.data_classes.NNParameters import NNLayerParameters
from ml_practice_and_neural_net.ml_practice.neural_nets.nn_activations import activation_map
from ml_practice_and_neural_net.ml_practice.weight_initializers import initializers_map
import numpy as np


class Layer:
    # This is for a fully connected layer in a NN
    # Needs a feed forward method. This method should receive a vector representing the previous output,
    # use its weights and biases to compute the next output, perform an activation, store output and activation for backprop, and return the output vector
        # weights are output_dim by input_dim, biases are output_dim by 1, compute W @ input + b
    # backprop method takes in the gradient of the loss with respect to the output of this layer, and computes the gradient of the loss with respect to the input of this layer, as well as the gradients with respect to weights and biases for updating
    # derivation: del_c/del_w = del_c/del_a (which is del_c/del_out-1) * del_a/del_z (which is activation_derivative wrt pre-activation output) * del_z/del_w (which is input)
    # del_c/del_b is the same but del_z/del_b is just 1, and del_c/del_input is del_c/del_a * del_a/del_z * del_z/del_input (which is weights)
    def __init__(self, params: NNLayerParameters):
        # width is also the output dim for a dense layer
        self.width = self.output_dim = params.width

        # this comes from the previous layer width (or if sparse, the number of neurons forwarded). Will be handled by neural net init
        self.input_dim = params.input_dim

        # self-explanatory
        self.activation, self.activation_derivative = activation_map[params.activation]()

        # weights = output by input, biases = output by 1
        self.weights = initializers_map[params.weight_init](output_dim = self.output_dim, input_dim = self.input_dim)
        self.biases = np.zeros(shape=(self.output_dim,))

        # pre-activation output, post-activation output, and input from previous iteration
        self.z = None
        self.a = None
        self.input = None

        # sparsity, only if defined. If not defined, layer is assumed to be dense and fully connected.
        self.sparsity = params.sparsity
        self.sparsity_mask = (np.random.rand(self.output_dim, self.input_dim) < self.sparsity) if self.sparsity < 1.0 else None


    def feed_forward(self, input_vector: NDArray) -> NDArray:
        self.input = input_vector
        self.z = (self.weights @ input_vector + self.biases) if not self.sparsity_mask else (self.weights * self.sparsity_mask) @ input_vector + self.biases
        self.a = self.activation(self.z)
        return self.a

    def backprop(self, prev_layer_gradient: NDArray) -> NDArray:
        # here, we have dc/da_1, need to take that and multiply by da_1/dz_0 * dz_0/dw, dz_0/db, and dz_0/dinput
        activation_derivative = self.activation_derivative(self.z) # da/dz

        # outer product. We want each weight vector to be changed by a scalar times input. Outer product makes a row representing the first scalar in our update times our first weight vector, etc
        weight_update = np.outer(prev_layer_gradient * activation_derivative, self.input)
        if self.sparsity_mask is not None:
            weight_update *= self.sparsity_mask

        bias_update = prev_layer_gradient * activation_derivative

        effective_weight_update = weight_update * self.sparsity
        input_gradient = (prev_layer_gradient * activation_derivative) @ effective_weight_update

        self.weights -= weight_update
        self.biases -= bias_update
        return input_gradient



