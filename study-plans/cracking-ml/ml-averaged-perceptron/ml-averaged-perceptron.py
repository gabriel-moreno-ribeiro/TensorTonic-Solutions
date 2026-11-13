
import numpy as np

def averaged_perceptron(X: list, y: list, lr: float, epochs: int) -> dict:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    
    w_sum = np.zeros(d)
    b_sum = 0.0
    count = 0
    
    for _ in range(epochs):
        for i in range(n):
            if y[i] * (X[i] @ w + b) <= 0:
                w += lr * y[i] * X[i]
                b += lr * y[i]
            # accumulate after every sample visit
            w_sum += w
            b_sum += b
            count += 1
    
    w_avg = w_sum / count
    b_avg = float(b_sum / count)
    
    test_scores = X @ w_avg + b_avg
    preds = np.where(test_scores > 0, 1, -1).tolist()
    
    return {
        "weights": [round(float(v), 4) for v in w_avg.tolist()],
        "bias": round(b_avg, 4),
        "predictions": [int(p) for p in preds]
    }
