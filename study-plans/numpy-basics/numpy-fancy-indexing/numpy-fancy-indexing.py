import numpy as np

def select_by_index(arr: list, indices: list, axis: int) -> np.ndarray:
    a = np.array(arr, dtype=np.float64)
    idx = np.array(indices, dtype=np.int64).reshape(-1)
    return np.take(a, idx, axis=int(axis))
