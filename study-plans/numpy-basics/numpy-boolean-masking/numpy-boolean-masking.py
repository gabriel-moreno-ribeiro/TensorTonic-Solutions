import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    a = np.array(data, dtype=np.float64)
    gt = a > threshold
    mask = gt.astype(np.float64)
    any_rows = gt.any(axis=1, keepdims=True)
    all_rows = gt.all(axis=1, keepdims=True)
    any_f = np.where(any_rows, a, 0.0)
    all_f = np.where(all_rows, a, 0.0)
    return np.stack([mask, any_f, all_f]).astype(np.float64)
