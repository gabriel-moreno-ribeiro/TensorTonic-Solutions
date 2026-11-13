import numpy as np

def norm_gate(X: list, W: list, threshold: float) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    W = np.array(W, dtype=np.float64)  
    Z = X @ W
    row_norms = np.linalg.norm(Z, axis=1)
    mask = (row_norms >= threshold)
    Y = np.where(mask[:, np.newaxis], Z, 0)
    return Y  
