import numpy as np

def solve_linear_system(A: list, b: list) -> np.ndarray:
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    return np.linalg.solve(A, b)
