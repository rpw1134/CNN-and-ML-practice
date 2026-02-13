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
    model_name: str = "linear"