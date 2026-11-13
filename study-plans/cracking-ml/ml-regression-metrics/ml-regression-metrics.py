import numpy as np

def regression_metrics(y_true: list, y_pred: list) -> dict:
    y = np.array(y_true, dtype=np.float64)
    yh = np.array(y_pred, dtype=np.float64)
    res = y - yh
    mse = round(float(np.mean(res**2)), 4)
    mae = round(float(np.mean(np.abs(res))), 4)
    tss = float(np.sum((y - y.mean())**2))
    r2 = 0.0 if tss == 0.0 else round(float(1 - np.sum(res**2) / tss), 4)
    return {"mse": mse, "mae": mae, "r2": r2}
