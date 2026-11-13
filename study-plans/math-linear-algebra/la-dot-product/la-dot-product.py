import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x = np.array(x)
    y = np.array(y)
    return float(np.dot(x, y))
