import numpy as np

def kmeans(X, k, max_iters=100, seed=42):
    X = np.array(X, dtype=np.float64)
    n = len(X)
    rng = np.random.RandomState(seed)
    init_idx = rng.choice(n, k, replace=False)
    centroids = X[init_idx].copy()
    for _ in range(max_iters):
        dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_c = centroids.copy()
        for j in range(k):
            mask = labels == j
            if mask.any(): new_c[j] = X[mask].mean(axis=0)
        if np.allclose(centroids, new_c): break
        centroids = new_c
    dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
    labels = np.argmin(dists, axis=1)
    return labels.tolist(), [[round(v, 4) for v in c] for c in centroids]
