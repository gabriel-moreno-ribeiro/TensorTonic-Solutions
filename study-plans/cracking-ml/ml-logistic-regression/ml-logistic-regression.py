
import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(n_iters):
        z = X @ w + b
        # bug: no clip, may overflow on edge cases
        p = 1 / (1 + np.exp(-z))
        err = p - y
        dw = X.T @ err / n
        db = np.sum(err) / n
        w -= lr * dw
        b -= lr * db

    return [round(x, 4) for x in w.tolist()], round(b, 4)
