import numpy as np

def matrix_trace(A: list) -> float:
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    return float(sum(A[i, i] for i in range(n)))
