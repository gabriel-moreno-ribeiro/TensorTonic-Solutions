import numpy as np

def cholesky_decompose(A: list):
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        s = A[j, j] - np.dot(L[j, :j], L[j, :j])
        if s <= 0:
            return None
        L[j, j] = np.sqrt(s)
        for i in range(j + 1, n):
            L[i, j] = (A[i, j] - np.dot(L[i, :j], L[j, :j])) / L[j, j]
    return L
