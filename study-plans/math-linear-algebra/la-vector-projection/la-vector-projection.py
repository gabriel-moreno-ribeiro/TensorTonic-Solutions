import numpy as np

def vector_projection(u: list, v: list) -> np.ndarray:
    u = np.array(u, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    return (np.dot(u, v) / np.dot(v, v)) * v
