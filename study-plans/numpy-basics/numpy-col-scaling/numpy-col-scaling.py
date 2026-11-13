import numpy as np

def scale_cols(data: list, weights: list) -> np.ndarray:
    a = np.array(data, dtype=np.float64)
    w = np.array(weights, dtype=np.float64).reshape(1, -1)
    return a * w
