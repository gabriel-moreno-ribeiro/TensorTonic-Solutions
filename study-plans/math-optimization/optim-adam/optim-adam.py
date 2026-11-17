import numpy as np

def adam(X: list, y: list, lr: float, beta1: float, beta2: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    losses = []
    for t in range(1, n_epochs + 1):
        losses.append(float(np.mean((X @ w - y)**2)))
        g = (2.0/n) * X.T @ (X @ w - y)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return {'losses': losses, 'final_weights': [float(x) for x in w]}
