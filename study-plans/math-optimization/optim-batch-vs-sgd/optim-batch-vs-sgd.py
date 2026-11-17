import numpy as np

def batch_gd_compare(X: list, y: list, batch_sizes: list, n_epochs: int, lr: float, seed: int) -> list:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n = len(X)
    results = []
    for bs in batch_sizes:
        rng = np.random.RandomState(seed)
        w = np.zeros(X.shape[1])
        losses = []
        for epoch in range(n_epochs):
            indices = rng.permutation(n)
            for start in range(0, n, bs):
                idx = indices[start:start+bs]
                Xb = X[idx]
                yb = y[idx]
                B = len(Xb)
                g = (2.0 / B) * Xb.T @ (Xb @ w - yb)
                w = w - lr * g
            mse = np.mean((X @ w - y)**2)
            losses.append(float(mse))
        results.append(losses)
    return results
