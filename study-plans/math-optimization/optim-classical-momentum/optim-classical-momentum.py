import numpy as np

def momentum_gd(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n = len(X)
    w_van = np.zeros(X.shape[1])
    w_mom = np.zeros(X.shape[1])
    v = np.zeros(X.shape[1])
    vanilla_losses = []
    momentum_losses = []
    for epoch in range(n_epochs):
        # record BEFORE update
        vanilla_losses.append(float(np.mean((X @ w_van - y)**2)))
        g_van = (2.0/n) * X.T @ (X @ w_van - y)
        w_van = w_van - lr * g_van
        
        momentum_losses.append(float(np.mean((X @ w_mom - y)**2)))
        g_mom = (2.0/n) * X.T @ (X @ w_mom - y)
        v = beta * v + g_mom
        w_mom = w_mom - lr * v
    return {'vanilla_losses': vanilla_losses, 'momentum_losses': momentum_losses}
