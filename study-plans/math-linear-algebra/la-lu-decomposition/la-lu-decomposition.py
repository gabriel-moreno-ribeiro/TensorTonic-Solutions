import numpy as np

def lu_decomposition(A: list) -> dict:
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = A.copy()
    for j in range(n):
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i] -= L[i, j] * U[j]
    return {"L": L, "U": U}
