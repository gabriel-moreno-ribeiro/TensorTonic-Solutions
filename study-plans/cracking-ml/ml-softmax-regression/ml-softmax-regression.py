
import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    n, d = X.shape
    K = num_classes
    W = np.zeros((d, K))
    b = np.zeros(K)

    for _ in range(n_iters):
        Z = X @ W + b
        Z -= Z.max(axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        P = exp_Z / exp_Z.sum(axis=1, keepdims=True)
        
        # one-hot encode y
        Y_oh = np.zeros_like(P)
        Y_oh[np.arange(n), y] = 1.0
        
        dP = (P - Y_oh) / n
        dW = X.T @ dP
        db = dP.sum(axis=0)
        W -= lr * dW
        b -= lr * db

    return [[round(v, 4) for v in row] for row in W.tolist()], [round(v, 4) for v in b.tolist()]
