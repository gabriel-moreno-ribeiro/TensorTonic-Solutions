import numpy as np

def winsorize(data: list, lo_q: float, hi_q: float) -> np.ndarray:  
    a = np.array(data, dtype=np.float64)
    lo, hi = np.percentile(a, [lo_q, hi_q], axis=0, method='linear') 
    clipped = np.clip(a, lo, hi)
    lo_mask = (a < lo).astype(float) 
    hi_mask = (a > hi).astype(float)
    return np.stack([clipped, lo_mask, hi_mask])
