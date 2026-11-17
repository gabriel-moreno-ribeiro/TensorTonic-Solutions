def convex_set_membership(A: list[list[float]], b: list[float], x: list[float]) -> dict:
    residuals = [sum(A[i][j]*x[j] for j in range(len(x))) - b[i] for i in range(len(A))]
    max_viol = max(residuals)
    in_set = bool(max_viol <= 1e-6)
    return {"in_set": in_set, "max_violation": round(max_viol, 6)}