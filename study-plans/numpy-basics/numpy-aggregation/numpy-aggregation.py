import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    a = np.array(data, dtype=np.float64)
    return np.stack([a.mean(axis), a.std(axis, ddof=0), a.min(axis), a.max(axis)])
