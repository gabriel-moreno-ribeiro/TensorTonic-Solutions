import numpy as np

def lasso_ista(X: list, y: list, lam: float, n_iters: int) -> list:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    XtX = X.T @ X
    max_eig = np.max(np.linalg.eigvalsh(XtX))
    L = max_eig / n
    step = 1.0 / L
    threshold = lam / L
    
    w = np.zeros(d)
    trajectory = [[float(x) for x in w]]
    
    for i in range(n_iters):
        grad = (1.0/n) * X.T @ (X @ w - y)
        z = w - step * grad
        w = np.sign(z) * np.maximum(np.abs(z) - threshold, 0)
        trajectory.append([float(x) for x in w])
    
    return trajectory
