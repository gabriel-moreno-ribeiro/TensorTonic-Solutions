import numpy as np

def low_rank_approximation(A: list, r: int) -> np.ndarray:
    A = np.array(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :r] @ np.diag(s[:r]) @ Vt[:r, :]
