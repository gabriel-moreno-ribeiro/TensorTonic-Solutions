import numpy as np

def compare_correlations(a: list, b: list) -> np.ndarray:
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.concatenate([a, b], axis=0)
    r_a = np.corrcoef(a.T)
    r_b = np.corrcoef(b.T) 
    r_c = np.corrcoef(c.T)
    return np.stack([r_a, r_b, r_c])
