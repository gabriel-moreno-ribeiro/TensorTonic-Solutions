import numpy as np

def hadamard_product(A: list, B: list) -> np.ndarray:
    return np.array(A, dtype=np.float64) * np.array(B, dtype=np.float64)
