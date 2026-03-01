import numpy as np
from typing import Tuple
from numpy.typing import NDArray

def add_data_bias_term(data: NDArray):
    """
    Adds bias 1s to a given dataset for bias purposes.
    :param data: np.NDArray: The data to which the bias term should be added. It is expected to be a 2D array where rows are samples and columns are features.
    :return: np.NDArray: A new array with a bias term (a column of ones) added as the first column.
    """
    return np.column_stack((np.ones(data.shape[0]), data))

def shuffle_data(data: NDArray, labels: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Shuffles the data and labels in unison to maintain the correct pairing between samples and their corresponding labels.
    :param data: np.NDArray: The dataset to be shuffled. It is expected to be a 2D array where rows are samples and columns are features.
    :param labels: np.NDArray: The corresponding labels for the dataset. It is expected to be an array or matrix where each element corresponds to the label of the respective sample in the data.
    :return: Tuple[np.NDArray, np.NDArray]: A tuple containing the shuffled data and shuffled labels in the following order: (shuffled_data, shuffled_labels).
    """
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    return data[indices], labels[indices]

def split_data(data: NDArray, labels: NDArray, test_ratio: float = 0.2, shuffle: bool = True)-> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
        Splits the data and labels into training and testing sets based on the specified ratio. Optionally shuffles the data before splitting.
    :param data: np.NDArray: The dataset to be split. It is expected to be a 2D array where rows are samples and columns are features.
    :param labels: np.NDArray: The corresponding labels for the dataset. It is expected to be an array or matrix where each element corresponds to the label of the respective sample in the data.
    :param test_ratio: float: The ratio of the dataset to be used for testing. The remaining portion will be used for testing. Default is 0.2
    :param shuffle: bool: Whether to shuffle the data and labels before splitting. Default is True.
    :return: Tuple[np.NDArray, np.NDArray, np.NDArray, np.NDArray]: A tuple containing the training data, training labels, testing data, and testing labels in the following order: (train_data, train_labels, test_data, test_labels).
    """
    indices = np.arange(data.shape[0])
    if shuffle:
        np.random.shuffle(indices)
    split_index = int(len(indices) * (1 - test_ratio))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return data[train_indices], labels[train_indices], data[test_indices], labels[test_indices]