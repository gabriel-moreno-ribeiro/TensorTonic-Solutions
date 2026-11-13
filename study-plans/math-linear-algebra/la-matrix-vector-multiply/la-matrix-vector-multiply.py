import numpy as np

def matrix_vector_multiply(A: list, x: list) -> np.ndarray:
    A = np.array(A, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    return A @ x
