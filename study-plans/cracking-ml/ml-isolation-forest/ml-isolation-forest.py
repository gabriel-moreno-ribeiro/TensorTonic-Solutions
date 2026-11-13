import numpy as np
import math

def isolation_forest(X: list, n_estimators: int = 100, max_samples: int = 256, seed: int = 42) -> list:
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.RandomState(seed)
    sample_size = min(max_samples, n)
    max_depth = math.ceil(math.log2(sample_size))
    
    def c(s):
        if s <= 1: return 0.0
        if s == 2: return 1.0
        return 2*(math.log(s-1) + 0.5772156649) - 2*(s-1)/s
    
    def build_tree(pts, depth):
        s = len(pts)
        if depth >= max_depth or s <= 1:
            return s
        varying = [j for j in range(d) if pts[:, j].min() < pts[:, j].max()]
        if not varying:
            return s
        j = int(rng.choice(varying))
        lo, hi = pts[:, j].min(), pts[:, j].max()
        t = rng.uniform(lo, hi)
        left = pts[pts[:, j] < t]
        right = pts[pts[:, j] >= t]
        return {'j': j, 't': t, 'l': build_tree(left, depth+1), 'r': build_tree(right, depth+1)}
    
    def path_length(node, x, depth):
        if isinstance(node, int):
            return depth + c(node)
        if x[node['j']] < node['t']:
            return path_length(node['l'], x, depth+1)
        return path_length(node['r'], x, depth+1)
    
    trees = []
    for _ in range(n_estimators):
        idx = rng.choice(n, sample_size, replace=False)
        trees.append(build_tree(X[idx], 0))
    
    scores = []
    for x in X:
        avg_h = np.mean([path_length(t, x, 0) for t in trees])
        scores.append(round(float(2 ** (-avg_h / c(sample_size))), 4))
    return scores
