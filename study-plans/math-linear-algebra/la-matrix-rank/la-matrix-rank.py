import numpy as np

def matrix_rank(A: list) -> int:
    return int(np.linalg.matrix_rank(np.array(A, dtype=np.float64)))
