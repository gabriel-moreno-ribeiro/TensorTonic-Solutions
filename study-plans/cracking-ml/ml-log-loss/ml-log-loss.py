import numpy as np

def log_loss(y_true: list, y_pred: list) -> float:
    y = np.array(y_true, dtype=np.float64)
    p = np.array(y_pred, dtype=np.float64)
    eps = 1e-15
    p = np.where(p < eps, eps, np.where(p > 1 - eps, 1 - eps, p))
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return round(float(np.mean(loss)), 4)
