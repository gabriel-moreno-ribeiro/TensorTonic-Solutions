
import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X @ w + b
        err = y_hat - y
        # The problem says "Add 2*alpha*w to the weight gradient"
        # but maybe the MSE part uses 2/n (not 1/n)
        dw = (2 / n) * X.T @ err + 2 * alpha * w
        db = (2 / n) * np.sum(err)
        w -= lr * dw
        b -= lr * db

    return [round(x, 4) for x in w.tolist()], round(b, 4)
