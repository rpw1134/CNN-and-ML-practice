import numpy as np

def gradient_descent(init_params, learning_rate, gradient_func, epsilon=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs a single step of gradient descent optimization.
    :param init_params: Initial parameters
    :param learning_rate: float: The learning rate for the gradient descent. It controls how much the weights are updated in each step.
    :param gradient_func: Callable[[np.NDArray], np.NDArray]: A function that takes the current weights as input and returns the gradient of the loss function with respect to those weights. The returned gradient should be a 1D array of the same shape as the weights.
    :param epsilon: float: A small constant representing the termination condition.
    :param num_iterations: int: Number of iterations to run the optimization for.
    :param regularization_gradient: Optional[Callable[[np.NDArray], np.NDArray]]: An optional function that computes the gradient of the regularization term with respect to the weights.
    :return: np.NDArray: The updated weights after performing one step of gradient descent.
    """
    params = init_params
    new_params = None
    if not regularization_gradient:
        regularization_gradient = lambda p: 0
    for i in range (num_iterations):
        new_grad = gradient_func(params)
        reg_component = regularization_gradient(params)
        new_params = params - learning_rate * (new_grad + reg_component)
        if np.linalg.norm(new_params - params) < epsilon:
            break
        params = new_params
    return new_params
