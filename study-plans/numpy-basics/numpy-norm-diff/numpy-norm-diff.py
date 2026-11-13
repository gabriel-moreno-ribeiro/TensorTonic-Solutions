import numpy as np

def norm_diff(a: list, b: list, lo: float, hi: float) -> np.ndarray:
    a = np.clip(a, lo, hi)  
    b = np.clip(b, lo, hi)
    return np.abs(a - b) / (hi - lo)
