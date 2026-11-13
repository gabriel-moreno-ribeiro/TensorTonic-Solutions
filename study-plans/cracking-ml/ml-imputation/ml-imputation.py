import numpy as np

def impute(X: list, method: str = "mean") -> list:
    arr = np.array(X, dtype=np.float64)
    result = arr.copy()
    for j in range(arr.shape[1]):
        col = arr[:, j]
        obs = col[~np.isnan(col)]
        if len(obs) == 0:
            fill = 0.0
        elif method == "mean":
            fill = float(np.mean(obs))
        else:
            fill = float(np.median(obs))
        result[np.isnan(col), j] = fill
    return [[round(float(v), 4) for v in row] for row in result]
