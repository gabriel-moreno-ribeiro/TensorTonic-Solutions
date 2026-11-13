import numpy as np

def linear_combination(vectors: list, coefficients: list) -> np.ndarray:
    V = np.array(vectors, dtype=np.float64)
    c = np.array(coefficients, dtype=np.float64)
    return c @ V
