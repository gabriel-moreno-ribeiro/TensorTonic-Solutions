import numpy as np

def adagrad(X: list, y: list, lr: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    G = np.zeros(d)
    losses = []
    effective_lrs = []
    for epoch in range(n_epochs):
        # record MSE before update
        losses.append(float(np.mean((X @ w - y)**2)))
        g = (2.0/n) * X.T @ (X @ w - y)
        G += g**2
        eff_lr = lr / np.sqrt(G + 1e-8)
        effective_lrs.append([float(e) for e in eff_lr])
        w = w - eff_lr * g
    return {'losses': losses, 'effective_lrs': effective_lrs}
