import numpy as np

def qr_decompose(A: list) -> dict:
    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    V = A.copy()
    for i in range(n):
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], V[:, i])
            V[:, i] -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(V[:, i])
        if R[i, i] < 0:
            R[i, i] = -R[i, i]
            V[:, i] = -V[:, i]
        Q[:, i] = V[:, i] / R[i, i]
    return {"Q": Q, "R": R}
