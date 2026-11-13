import numpy as np

def kmeans_plusplus(X: list, k: int, seed: int = 42) -> list:
    X = np.array(X, dtype=np.float64)
    rng = np.random.RandomState(seed)
    n = len(X)
    
    first = rng.randint(0, n)
    centroids = [X[first]]
    
    for _ in range(k - 1):
        # Vectorized D^2
        C = np.array(centroids)  # shape (m, d)
        # dists[i] = min over centroids of squared distance
        diffs = X[:, None, :] - C[None, :, :]  # (n, m, d)
        sq = np.sum(diffs**2, axis=2)  # (n, m)
        dists = sq.min(axis=1)  # (n,)
        probs = dists / dists.sum()
        idx = rng.choice(n, p=probs)
        centroids.append(X[idx])
    
    return [[round(float(v), 4) for v in c] for c in centroids]
