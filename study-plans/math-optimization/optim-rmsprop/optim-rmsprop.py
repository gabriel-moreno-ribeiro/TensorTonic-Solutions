import numpy as np

def rmsprop(X: list, y: list, lr: float, decay: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    v = np.zeros(d)
    losses = []
    effective_lrs = []
    for epoch in range(n_epochs):
        losses.append(float(np.mean((X @ w - y)**2)))
        g = (2.0/n) * X.T @ (X @ w - y)
        v = decay * v + (1 - decay) * g**2
        eff_lr = lr / np.sqrt(v + 1e-8)
        effective_lrs.append([float(e) for e in eff_lr])
        w = w - eff_lr * g
    return {'losses': losses, 'effective_lrs': effective_lrs}
