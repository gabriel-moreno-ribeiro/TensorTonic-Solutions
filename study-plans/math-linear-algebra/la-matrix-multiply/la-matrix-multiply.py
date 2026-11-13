import numpy as np

def matrix_multiply(A: list, B: list) -> np.ndarray:
    return np.array(A, dtype=np.float64) @ np.array(B, dtype=np.float64)
