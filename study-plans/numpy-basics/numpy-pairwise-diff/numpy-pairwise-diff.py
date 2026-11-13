import numpy as np

def pairwise_diff(a: list) -> np.ndarray:
    x = np.array(a, dtype=np.float64).reshape(-1, 1)
    return x - x.T
