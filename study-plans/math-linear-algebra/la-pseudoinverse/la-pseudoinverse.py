import numpy as np

def pseudoinverse(A: list) -> np.ndarray:
    A = np.array(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = 1e-10
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T
