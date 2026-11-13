import numpy as np

def outer_sum(a: list, b: list) -> np.ndarray:
    x = np.array(a, dtype=np.float64).reshape(-1, 1)
    y = np.array(b, dtype=np.float64).reshape(1, -1)
    return x + y
