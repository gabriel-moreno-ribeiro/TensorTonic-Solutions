import numpy as np

def linear_warmup(X: list, y: list, base_lr: float, warmup_epochs: int, total_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    lr_schedule = []
    losses = []
    for epoch in range(1, total_epochs + 1):
        if epoch <= warmup_epochs:
            lr = base_lr * epoch / warmup_epochs
        else:
            lr = base_lr
        lr_schedule.append(float(lr))
        losses.append(float(np.mean((X @ w - y)**2)))
        g = (2.0/n) * X.T @ (X @ w - y)
        w = w - lr * g
    return {'lr_schedule': lr_schedule, 'losses': losses}
