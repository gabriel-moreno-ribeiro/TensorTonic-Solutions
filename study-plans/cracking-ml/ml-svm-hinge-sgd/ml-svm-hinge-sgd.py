
import numpy as np

def svm_hinge_sgd(X: list, y: list, lr: float, lam: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    
    for _ in range(n_epochs):
        for i in range(n):
            margin = y[i] * (X[i] @ w + b)
            if margin < 1:
                w = w - lr * (lam * w - y[i] * X[i])
                b = b + lr * y[i]
            else:
                w = w - lr * lam * w
    
    scores = X @ w + b
    preds = [1 if s > 0 else -1 for s in scores.tolist()]
    
    return {
        "weights": [float(round(v, 4)) for v in w.tolist()],
        "bias": float(round(b, 4)),
        "predictions": preds
    }
