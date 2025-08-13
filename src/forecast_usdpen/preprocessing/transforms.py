import numpy as np


def create_sequences(data: np.ndarray, seq_length: int):
    """
    Creates sequences for time series forecasting with LSTM-like models.

    Args:
        data (np.ndarray): The input data array (scaled).
        seq_length (int): The length of the sequences.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of X (sequences) and y (targets).
    """
    X, y = [], []
    data_array = data.reshape(-1, 1) if len(data.shape) == 1 else data
    for i in range(len(data_array) - seq_length):
        X.append(data_array[i : (i + seq_length)])
        y.append(data_array[i + seq_length])
    return np.array(X), np.array(y)
