import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    a = np.array(data, dtype=np.float64)
    sub = a[row_start:row_stop]
    out = sub[sub > threshold]
    return np.asarray(out, dtype=np.float64).reshape(-1)
