import numpy as np

def gram_schmidt(vectors: list) -> np.ndarray:
    V = np.array(vectors, dtype=np.float64)
    k, n = V.shape
    Q = np.zeros((k, n), dtype=np.float64)
    for i in range(k):
        r = V[i].copy()
        for j in range(i):
            r -= np.dot(r, Q[j]) * Q[j]
        Q[i] = r / np.linalg.norm(r)
    return Q
