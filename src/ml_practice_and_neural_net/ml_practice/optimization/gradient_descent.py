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

# For implementation and study purposes
def momentum_optimizer(init_params, learning_rate, gradient_func, beta=0.9, epsilon=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs Momentum optimization.

    :param init_params: Initial parameters
    :param learning_rate: float: The learning rate (alpha). Typical values: 0.01, 0.001
    :param gradient_func: Callable: Function that computes the gradient
    :param beta: float: Momentum factor (exponential decay rate). Default: 0.9
    :param epsilon: float: Small constant for convergence check. Default: 1e-8
    :param num_iterations: int: Number of iterations
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    params = init_params
    velocity = np.zeros_like(params)  # Initialize velocity/momentum term

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    for i in range(num_iterations):
        # Compute total gradient (loss + regularization)
        grad = gradient_func(params)
        reg_component = regularization_gradient(params)
        total_grad = (1 - beta) * (grad + reg_component)

        # Update VELOCITY with momentum
        velocity = beta * velocity + total_grad

        # Update parameters
        new_params = params - learning_rate * velocity

        # Check convergence
        if np.linalg.norm(new_params - params) < epsilon:
            break
        params = new_params

    return params


def adam_optimizer(init_params, learning_rate, gradient_func, beta=0.9, gamma=0.999, epsilon=1e-8, termination_difference=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs Adam (Adaptive Moment Estimation) optimization.

    :param init_params: Initial parameters
    :param learning_rate: float: The learning rate (alpha). Typical values: 0.001, 0.0001
    :param gradient_func: Callable: Function that computes the gradient
    :param beta: float: Exponential decay rate for first moment estimates. Default: 0.9
    :param gamma: float: Exponential decay rate for second moment estimates. Default: 0.999
    :param epsilon: float: Small constant for numerical stability. Default: 1e-8
    :param termination_difference: float: Threshold for convergence. Default: 1e-8
    :param num_iterations: int: Number of iterations
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    params = init_params
    sum_squares_av = np.zeros_like(params)  # Initialize velocity/momentum term
    velocity = np.zeros_like(params)

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    for i in range(num_iterations):
        # Compute total gradient (loss + regularization)
        grad = gradient_func(params)
        reg_component = regularization_gradient(params)
        total_grad = grad + reg_component

        # Update biased first moment (momentum) and second moment (squared gradient moving average)
        velocity = beta * velocity + (1 - beta) * total_grad
        sum_squares_av = gamma * sum_squares_av + (1 - gamma) * (total_grad ** 2)

        # Bias correction for moments (corrects initialization bias, especially important in early iterations)
        m_hat = velocity / (1 - beta ** (i + 1))
        v_hat = sum_squares_av / (1 - gamma ** (i + 1))

        # Update parameters with adaptive learning rate
        new_params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Check convergence
        if np.linalg.norm(new_params - params) < termination_difference:
            break
        params = new_params

    return params


def adam_mini_batch(init_params, X, y, learning_rate, loss_builder, batch_size=32, beta=0.9, gamma=0.999, epsilon=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs Adam optimization with mini-batch gradient descent.

    :param init_params: Initial parameters
    :param X: Training features (NxD)
    :param y: Training labels
    :param learning_rate: float: The learning rate. Typical values: 0.001, 0.0001
    :param loss_builder: Function that builds loss and gradient functions from data
    :param batch_size: int: Size of mini-batches. Default: 32
    :param beta: float: Exponential decay rate for first moment. Default: 0.9
    :param gamma: float: Exponential decay rate for second moment. Default: 0.999
    :param epsilon: float: Small constant for numerical stability. Default: 1e-8
    :param num_iterations: int: Number of epochs
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    num_examples = X.shape[0]
    if batch_size is None or batch_size > num_examples:
        batch_size = num_examples

    params = init_params
    velocity = np.zeros_like(params)
    sum_squares_av = np.zeros_like(params)

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    iteration = 0  # Track global iteration for bias correction

    for epoch in range(num_iterations):
        # Shuffle data at start of each epoch
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        curr_num_examples = 0
        while curr_num_examples < num_examples:
            iteration += 1

            # Get batch indices
            if num_examples - curr_num_examples < batch_size:
                batch_indices = indices[curr_num_examples:]
            else:
                batch_indices = indices[curr_num_examples:curr_num_examples + batch_size]

            curr_num_examples += len(batch_indices)

            # Compute gradient on batch
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            _, grad_func = loss_builder(batch_X, batch_y)
            grad = grad_func(params)
            reg_component = regularization_gradient(params)
            total_grad = grad + reg_component

            # Update biased first moment (momentum) and second moment (squared gradient moving average)
            velocity = beta * velocity + (1 - beta) * total_grad
            sum_squares_av = gamma * sum_squares_av + (1 - gamma) * (total_grad ** 2)

            # Bias correction for moments
            m_hat = velocity / (1 - beta ** iteration)
            v_hat = sum_squares_av / (1 - gamma ** iteration)

            # Update parameters with adaptive learning rate
            params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return params


def adagrad_optimizer(init_params, learning_rate, gradient_func, epsilon=1e-8, termination_difference=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs Adagrad (Adaptive Gradient) optimization.

    :param init_params: Initial parameters
    :param learning_rate: float: Base learning rate. Typical values: 0.01, 0.001
    :param gradient_func: Callable: Function that computes the gradient
    :param epsilon: float: Small constant for numerical stability. Default: 1e-8
    :param termination_difference: float: Threshold for convergence. Default: 1e-8
    :param num_iterations: int: Number of iterations
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    params = init_params
    sum_squares = np.zeros_like(params)  # Initialize velocity/momentum term

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    for i in range(num_iterations):
        # Compute total gradient (loss + regularization)
        grad = gradient_func(params)
        reg_component = regularization_gradient(params)
        total_grad = grad + reg_component

        # Accumulate squared gradient
        sum_squares = sum_squares + total_grad ** 2

        # Update parameters with adaptive learning rate
        new_params = params - (learning_rate / np.sqrt(sum_squares + epsilon)) * total_grad

        # Check convergence
        if np.linalg.norm(new_params - params) < termination_difference:
            break
        params = new_params

    return params


def adagrad_mini_batch(init_params, X, y, learning_rate, loss_builder, batch_size=32, epsilon=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs Adagrad optimization with mini-batch gradient descent.

    :param init_params: Initial parameters
    :param X: Training features (NxD)
    :param y: Training labels
    :param learning_rate: float: Base learning rate. Typical values: 0.01, 0.001
    :param loss_builder: Function that builds loss and gradient functions from data
    :param batch_size: int: Size of mini-batches. Default: 32
    :param epsilon: float: Small constant for numerical stability. Default: 1e-8
    :param num_iterations: int: Number of epochs
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    num_examples = X.shape[0]
    if batch_size is None or batch_size > num_examples:
        batch_size = num_examples

    params = init_params
    sum_squares = np.zeros_like(params)

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    for epoch in range(num_iterations):
        # Shuffle data at start of each epoch
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        curr_num_examples = 0
        while curr_num_examples < num_examples:
            # Get batch indices
            if num_examples - curr_num_examples < batch_size:
                batch_indices = indices[curr_num_examples:]
            else:
                batch_indices = indices[curr_num_examples:curr_num_examples + batch_size]

            curr_num_examples += len(batch_indices)

            # Compute gradient on batch
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            _, grad_func = loss_builder(batch_X, batch_y)
            grad = grad_func(params)
            reg_component = regularization_gradient(params)
            total_grad = grad + reg_component

            # Accumulate squared gradient
            sum_squares = sum_squares + total_grad ** 2

            # Update parameters with adaptive learning rate
            params = params - (learning_rate / np.sqrt(sum_squares + epsilon)) * total_grad

    return params


def rmsprop_optimizer(init_params, learning_rate, gradient_func, gamma=0.9, epsilon=1e-8, termination_difference=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs RMSprop (Root Mean Square Propagation) optimization.

    :param init_params: Initial parameters
    :param learning_rate: float: Base learning rate. Typical values: 0.001, 0.0001
    :param gradient_func: Callable: Function that computes the gradient
    :param gamma: float: Decay rate for moving average of squared gradients. Default: 0.9
    :param epsilon: float: Small constant for numerical stability. Default: 1e-8
    :param termination_difference: float: Threshold for convergence. Default: 1e-8
    :param num_iterations: int: Number of iterations
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    params = init_params
    sum_squares_av = np.zeros_like(params)  # Initialize velocity/momentum term

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    for i in range(num_iterations):
        # Compute total gradient (loss + regularization)
        grad = gradient_func(params)
        reg_component = regularization_gradient(params)
        total_grad = grad + reg_component

        # Accumulate squared gradient
        sum_squares_av = gamma * sum_squares_av + (1 - gamma) * total_grad ** 2

        # Update parameters with adaptive learning rate
        new_params = params - (learning_rate / np.sqrt(sum_squares_av + epsilon)) * total_grad

        # Check convergence
        if np.linalg.norm(new_params - params) < termination_difference:
            break
        params = new_params

    return params


def rmsprop_mini_batch(init_params, X, y, learning_rate, loss_builder, batch_size=32, gamma=0.9, epsilon=1e-8, termination_difference=1e-8, num_iterations=1000, regularization_gradient=None):
    """
    Performs RMSprop optimization with mini-batch gradient descent.

    :param init_params: Initial parameters
    :param X: Training features (NxD)
    :param y: Training labels
    :param learning_rate: float: Base learning rate. Typical values: 0.001, 0.0001
    :param loss_builder: Function that builds loss and gradient functions from data
    :param batch_size: int: Size of mini-batches. Default: 32
    :param gamma: float: Decay rate for moving average of squared gradients. Default: 0.9
    :param epsilon: float: Small constant for numerical stability. Default: 1e-8
    :param termination_difference: float: Threshold for convergence. Default: 1e-8
    :param num_iterations: int: Number of epochs
    :param regularization_gradient: Optional regularization gradient function
    :return: np.NDArray: Optimized parameters
    """
    num_examples = X.shape[0]
    if batch_size is None or batch_size > num_examples:
        batch_size = num_examples

    params = init_params
    sum_squares_av = np.zeros_like(params)

    if not regularization_gradient:
        regularization_gradient = lambda p: np.zeros_like(p)

    for epoch in range(num_iterations):
        old_params = params.copy()

        # Shuffle data at start of each epoch
        indices = np.arange(num_examples)
        np.random.shuffle(indices)

        curr_num_examples = 0
        while curr_num_examples < num_examples:
            # Get batch indices
            if num_examples - curr_num_examples < batch_size:
                batch_indices = indices[curr_num_examples:]
            else:
                batch_indices = indices[curr_num_examples:curr_num_examples + batch_size]

            curr_num_examples += len(batch_indices)

            # Compute gradient on batch
            batch_X = X[batch_indices]
            batch_y = y[batch_indices]
            _, grad_func = loss_builder(batch_X, batch_y)
            grad = grad_func(params)
            reg_component = regularization_gradient(params)
            total_grad = grad + reg_component

            # Update moving average of squared gradient
            sum_squares_av = gamma * sum_squares_av + (1 - gamma) * (total_grad ** 2)

            # Update parameters with adaptive learning rate
            params = params - (learning_rate / np.sqrt(sum_squares_av + epsilon)) * total_grad

        # Check convergence after each epoch
        if np.linalg.norm(params - old_params) < termination_difference:
            break

    return params

