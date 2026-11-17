import numpy as np

def cosine_restarts(X: list, y: list, eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    t_cur = 0
    T_cur = T_0
    lr_schedule = []
    losses = []
    for epoch in range(total_epochs):
        lr = eta_min + 0.5*(eta_max - eta_min)*(1 + np.cos(np.pi * t_cur / T_cur))
        lr_schedule.append(float(lr))
        losses.append(float(np.mean((X @ w - y)**2)))
        g = (2.0/n) * X.T @ (X @ w - y)
        w = w - lr * g
        t_cur += 1
        if t_cur >= T_cur:
            t_cur = 0
            T_cur *= T_mult
    return {'lr_schedule': lr_schedule, 'losses': losses}
