import numpy as np

def matrix_determinant(A: list) -> float:
    return float(np.linalg.det(np.array(A, dtype=np.float64)))
