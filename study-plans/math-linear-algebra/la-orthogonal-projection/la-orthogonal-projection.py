import numpy as np

def projection_matrix(A: list) -> np.ndarray:
    A = np.array(A, dtype=np.float64)
    X = np.linalg.solve(A.T @ A, A.T)
    return A @ X
