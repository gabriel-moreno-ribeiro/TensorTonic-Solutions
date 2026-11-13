import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    difference = x_array - y_array
    return float(np.sqrt(np.sum(difference ** 2)))
    