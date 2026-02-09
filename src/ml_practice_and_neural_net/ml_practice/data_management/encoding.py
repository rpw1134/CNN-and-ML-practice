import numpy as np
from numpy.typing import NDArray

def normalize_categories(labels: NDArray, categories: NDArray) -> NDArray:
    """
    Normalizes a set of labels based on a provided mapping.
    :param labels: np.NDArray: An array of labels that need to be normalized. Each element in this array corresponds to a label for a sample in the dataset.
    :param categories: np.ndArray: An array of the unique categories present in the dataset. Each element in this array corresponds to a unique category that can be found in the labels array. The order of categories in this array will determine the mapping of original labels to normalized values.
    :return: np.NDArray: An array of normalized labels where each original label has been replaced by its corresponding normalized value as defined in the labels_map.
    """
    return np.searchsorted(categories, labels)

def convert_to_one_hot(labels: NDArray) -> NDArray:
    """
    Converts a set of labels to one-hot encoding based on a provided mapping.
    :param labels: np.NDArray: An array of labels that need to be converted to one-hot encoding. Each element in this array corresponds to a label for a sample in the dataset.
    :return: np.NDArray: A 2D array where each row corresponds to a one-hot encoded vector for the respective label in the input array. The number of columns will be equal to the number of unique labels as defined in the labels_map.
    """
    categories = np.unique(labels)
    normalized_categories = normalize_categories(labels, categories)
    one_hot = np.zeros((labels.shape[0], len(categories)), dtype=int)
    one_hot[np.arange(labels.shape[0]), normalized_categories] = 1
    return one_hot