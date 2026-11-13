import numpy as np

def log_loss(y_true: list, y_pred: list) -> float:
    y = np.array(y_true, dtype=np.float64)
    p = np.clip(np.array(y_pred, dtype=np.float64), 1e-15, 1 - 1e-15)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return round(float(loss.mean()), 4)
