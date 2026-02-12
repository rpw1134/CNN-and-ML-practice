from dataclasses import dataclass
from typing import Literal
from numpy.typing import NDArray

@dataclass
class ModelData:
    """
    A data class to encapsulate all necessary metadata required for training a machine learning model.
    This includes the input data, labels, chosen regularization technique, and optimization method.

    Hyperparameters such as learning rate, number of training iterations, and regularization strength are not included here, as they are defined separately in a Hyperparameters data class.
    """

    given_data: NDArray
    labels: NDArray
    regularizer: Literal["l1", "l2", "elastic_net"] | None           # The regularization technique to apply during training.
    training_method: Literal["gradient_descent", "minibatch"] | None # The optimization method. Will extend to Adam, RMSProp, etc. in the future.
