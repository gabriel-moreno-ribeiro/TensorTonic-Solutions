import numpy as np

def learning_rate_sweep(X: list, y: list, learning_rates: list, n_epochs: int) -> list:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    result = []
    for lr in learning_rates:
        w = np.zeros(d)
        losses = []
        for epoch in range(n_epochs):
            losses.append(float(np.mean((X @ w - y)**2)))
            g = (2.0/n) * X.T @ (X @ w - y)
            w = w - lr * g
        result.append(losses)
    return result
