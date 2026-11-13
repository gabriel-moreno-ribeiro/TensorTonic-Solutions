import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    a = np.array(data, dtype=np.float64)
    row = a[row_idx].copy()
    clipped = np.clip(row, lo, hi)
    return np.stack([row, clipped]).astype(np.float64)
