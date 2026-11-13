import numpy as np

def pca_project(X: list, n_components: int) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / (X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(C)
    idx = np.argsort(-vals)[:n_components]
    Vk = vecs[:, idx]
    return Xc @ Vk
