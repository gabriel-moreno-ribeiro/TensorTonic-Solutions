import numpy as np

def kmeans_plusplus(X: list, k: int, seed: int = 42) -> list:
    X = np.array(X, dtype=np.float64)
    rng = np.random.RandomState(seed)
    n = len(X)
    
    first = rng.randint(0, n)
    centroids = [X[first]]
    
    for _ in range(k - 1):
        # Recompute D^2 to nearest centroid
        dists = np.array([
            min(np.sum((x - c)**2) for c in centroids)
            for x in X
        ])
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids.append(X[idx])
    
    return [[round(float(v), 4) for v in c] for c in centroids]
