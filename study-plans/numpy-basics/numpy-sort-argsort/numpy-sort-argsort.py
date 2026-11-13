import numpy as np

def sort_with_indices(data: list, axis: int) -> np.ndarray:  
    a = np.array(data, dtype=np.float64)
    s = np.sort(a, axis=axis)
    i = np.argsort(a, axis=axis).astype(np.float64) 
    return np.stack([s, i])
