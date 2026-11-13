
import numpy as np

def whiten(X: list) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    mu = np.mean(X, axis=0)
    Xc = X - mu
    
    # sample covariance
    C = Xc.T @ Xc / (len(X) - 1)
    
    # eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # orient each eigenvector so largest-magnitude coord is positive
    for j in range(eigenvectors.shape[1]):
        v = eigenvectors[:, j]
        if v[np.argmax(np.abs(v))] < 0:
            eigenvectors[:, j] = -v
    
    # scale: zero out near-zero eigenvalues
    threshold = 1e-10
    scale = np.where(eigenvalues > threshold, 1.0 / np.sqrt(eigenvalues), 0.0)
    
    # whitening: Xc @ V @ diag(scale)
    Xw = Xc @ eigenvectors * scale
    
    return Xw
