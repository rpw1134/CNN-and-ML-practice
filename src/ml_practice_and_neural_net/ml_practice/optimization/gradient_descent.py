import numpy as np

def gradient_descent(init_params, learning_rate, gradient_func, epsilon=1e-8, num_iterations=1000, regularization_gradient=None, batch_size=None):
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

def mini_batch_gradient_descent(init_params, X, y, learning_rate, loss_builder, batch_size=None, epsilon=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs mini-batch gradient descent optimization.
    :param init_params: Initial parameters
    :param X: np.NDArray: The input data for which the loss function and gradient will be built. It is expected to be a 2D array where rows are samples and columns are features.
    :param y: np.NDArray: The target values corresponding to the input data.
    :param learning_rate: float: The learning rate for the gradient descent. It controls how much the weights are updated in each step.
    :param loss_builder: Callable[[np.NDArray, np.NDArray], Tuple[Callable[[np.NDArray], float], Callable[[np.NDArray], np.NDArray]]]: A function that takes the training data and labels as input and returns a tuple containing the loss function and the gradient function. The loss function takes a parameter array as input and returns a float representing the loss. The gradient function takes a parameter array as input and returns an array representing the gradient of the loss function with respect to the parameters.
    :param batch_size: int: The size of each mini-batch used for computing the gradient.
    :param epsilon: float: A small constant representing the termination condition.
    :param num_iterations: int: Number of iterations to run the optimization for.
    :param regularization_gradient: Optional[Callable[[np.NDArray], np.NDArray]]: An optional function that computes the gradient of the regularization term with respect to the weights.
    :return: np.NDArray: The updated weights after performing mini-batch gradient descent.
    """
    # Define the batch size to train on
    num_examples = X.shape[0]
    if batch_size is None or batch_size > num_examples:
        batch_size = num_examples

    params = init_params
    new_params = None

    if not regularization_gradient:
        regularization_gradient = lambda p: 0

    for i in range (num_iterations):
        # new iteration (epoch)
        curr_num_examples = 0
        old_params = params.copy()

        # shuffle the data (without mutating)
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        # while we still haven't seen our whole set
        while curr_num_examples < num_examples:
            # if the remaining examples are less than the batch size, we take all of them, otherwise we take a full batch
            if num_examples - curr_num_examples < batch_size:
                batch_indices = indices[curr_num_examples:]
            else:
                batch_indices = indices[curr_num_examples:curr_num_examples+batch_size]

            # update the number of examples we've seen so far
            curr_num_examples += len(batch_indices)

            # compute the gradient on the current batch and update the parameters
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            _, grad_func = loss_builder(batch_X, batch_y)
            new_grad = grad_func(params)
            reg_component = regularization_gradient(params)
            new_params = params - learning_rate * (new_grad + reg_component)
            params = new_params

        # after an epoch, if we made little progress, we can stop the optimization
        if np.linalg.norm(params - old_params) < epsilon:
                break
    return params
