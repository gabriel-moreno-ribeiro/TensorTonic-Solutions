import numpy as np

def pca(X: list, n_components: int = 2) -> tuple:
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (n - 1)
    
    vals, vecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Orient each eigenvector: make largest-abs-value loading positive
    for j in range(vecs.shape[1]):
        col = vecs[:, j]
        sign = np.sign(col[np.argmax(np.abs(col))])
        if sign < 0:
            vecs[:, j] *= -1
    
    # Project
    proj = Xc @ vecs[:, :n_components]
    ratio = vals[:n_components] / vals.sum()
    
    transformed = [[round(float(v), 4) for v in row] for row in proj]
    ratio_rounded = [round(float(r), 4) for r in ratio]
    return (transformed, ratio_rounded)
