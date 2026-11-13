import numpy as np

def create_filled_array(shape: list, kind: str) -> np.ndarray:
    shape = tuple(int(s) for s in shape)
    if kind == "ones":
        return np.ones(shape, dtype=np.float64)
    return np.zeros(shape, dtype=np.float64)
