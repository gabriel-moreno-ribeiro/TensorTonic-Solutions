import numpy as np

def mahalanobis_distance(x: list, mean: list, cov: list) -> float:
    x = np.array(x, dtype=np.float64)
    mu = np.array(mean, dtype=np.float64)
    S = np.array(cov, dtype=np.float64)
    diff = x - mu
    return float(np.sqrt(diff @ np.linalg.solve(S, diff)))
