import numpy as np

def tile_diff(data: list, reps: int) -> np.ndarray:
    a = np.array(data, dtype=np.float64)
    t = np.tile(a, (reps, 1))
    d = np.zeros_like(t)
    d[:-1] = np.diff(t, axis=0)
    return np.stack([t, d]) 
