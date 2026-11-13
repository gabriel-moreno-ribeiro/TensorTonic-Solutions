import numpy as np

def svd(A: list) -> dict:
    A = np.array(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return {"U": U, "s": s, "Vt": Vt}
