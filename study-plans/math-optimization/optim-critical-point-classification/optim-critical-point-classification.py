import numpy as np

def classify_critical_point(H: list[list[float]]) -> str:
    tol = 1e-6
    eigs = np.linalg.eigvalsh(np.array(H))
    pos = sum(1 for e in eigs if e > tol)
    neg = sum(1 for e in eigs if e < -tol)
    zero = len(eigs) - pos - neg
    if zero > 0: return "degenerate"
    if pos == len(eigs): return "local_min"
    if neg == len(eigs): return "local_max"
    return "saddle"