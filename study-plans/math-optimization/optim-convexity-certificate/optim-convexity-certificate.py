import numpy as np

def convexity_certificate(H: list[list[float]]) -> dict:
    mat = np.array(H)
    eigvals = np.linalg.eigvalsh(mat)
    min_eig = float(eigvals.min())
    is_convex = bool(min_eig >= -1e-6)
    return {"is_convex": is_convex, "min_eigenvalue": round(min_eig, 6)}