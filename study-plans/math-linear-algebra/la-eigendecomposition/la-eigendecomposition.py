import numpy as np

def eigendecompose(A: list) -> dict:
    A = np.array(A, dtype=np.float64)
    vals, vecs = np.linalg.eig(A)
    idx = np.argsort(-np.abs(vals))
    vals = vals[idx].real
    vecs = vecs[:, idx].real
    for i in range(vecs.shape[1]):
        if vecs[np.argmax(np.abs(vecs[:, i])), i] < 0:
            vecs[:, i] *= -1
    return {"eigenvalues": vals, "eigenvectors": vecs}
