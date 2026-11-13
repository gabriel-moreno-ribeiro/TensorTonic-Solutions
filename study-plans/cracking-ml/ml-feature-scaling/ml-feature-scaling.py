import numpy as np

def feature_scale(X: list, method: str = "minmax") -> list:
    X = np.array(X, dtype=np.float64)
    if method == "minmax":
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        denom = mx - mn
        denom[denom == 0] = 1.0
        result = (X - mn) / denom
    else:
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        sigma[sigma == 0] = 1.0
        result = (X - mu) / sigma
    return [[round(float(v), 4) for v in row] for row in result]
