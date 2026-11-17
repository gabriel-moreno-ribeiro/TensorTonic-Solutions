import numpy as np

def nesterov_momentum(X: list, y: list, lr: float, beta: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n = len(X)
    w_c = np.zeros(X.shape[1])
    w_n = np.zeros(X.shape[1])
    v_c = np.zeros(X.shape[1])
    v_n = np.zeros(X.shape[1])
    classical_losses = []
    nesterov_losses = []
    for epoch in range(n_epochs):
        classical_losses.append(float(np.mean((X @ w_c - y)**2)))
        g_c = (2.0/n) * X.T @ (X @ w_c - y)
        v_c = beta * v_c + g_c
        w_c = w_c - lr * v_c
        
        nesterov_losses.append(float(np.mean((X @ w_n - y)**2)))
        w_look = w_n - lr * beta * v_n
        g_n = (2.0/n) * X.T @ (X @ w_look - y)
        v_n = beta * v_n + g_n
        w_n = w_n - lr * v_n
    return {'classical_losses': classical_losses, 'nesterov_losses': nesterov_losses}
