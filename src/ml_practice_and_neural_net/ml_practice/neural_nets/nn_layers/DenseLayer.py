class DenseLayer:
    # This is for a fully connected layer in a NN
    # Needs a feed forward method. This method should receive a vector representing the previous output,
    # use its weights and biases to compute the next output, perform an activation, store output and activation for backprop, and return the output vector
        # weights are output_dim by input_dim, biases are output_dim by 1, compute W @ input + b
    # backprop method takes in the gradient of the loss with respect to the output of this layer, and computes the gradient of the loss with respect to the input of this layer, as well as the gradients with respect to weights and biases for updating
    # derivation: del_c/del_w = del_c/del_a (which is del_c/del_out-1) * del_a/del_z (which is activation_derivative wrt pre-activation output) * del_z/del_w (which is input)
    # del_c/del_b is the same but del_z/del_b is just 1, and del_c/del_input is del_c/del_a * del_a/del_z * del_z/del_input (which is weights)
    pass