import math

def distance_metric(x: list, y: list, metric: str, p: float = 2) -> float:
    n = len(x)
    diff = [x[i] - y[i] for i in range(n)]
    
    if metric == "euclidean":
        result = math.sqrt(sum(d**2 for d in diff))
    elif metric == "manhattan":
        result = sum(abs(d) for d in diff)
    elif metric == "chebyshev":
        result = max(abs(d) for d in diff)
    elif metric == "minkowski":
        result = sum(abs(d)**p for d in diff) ** (1/p)
    elif metric == "cosine":
        dot = sum(x[i]*y[i] for i in range(n))
        nx = math.sqrt(sum(v**2 for v in x))
        ny = math.sqrt(sum(v**2 for v in y))
        if nx == 0 or ny == 0:
            result = 0.0
        else:
            result = 1 - dot / (nx * ny)
    else:
        result = 0.0
    
    return round(float(result), 4)
