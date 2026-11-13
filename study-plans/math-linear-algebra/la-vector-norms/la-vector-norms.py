import numpy as np

def vector_norms(v: list) -> np.ndarray:
    """
    Returns a float64 array containing the L1, L2, and infinity norms.
    """
    vector = np.asarray(v, dtype=np.float64)
    l1_norm = np.sum(np.abs(vector))
    l2_norm = np.sqrt(np.sum(vector ** 2))
    infinity_norm = np.max(np.abs(vector))
    return np.array([l1_norm, l2_norm, infinity_norm], dtype=np.float64)