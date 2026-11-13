import numpy as np

def agglomerative(X: list, n_clusters: int = 2, linkage: str = "single") -> list:
    X = np.array(X, dtype=np.float64)
    n = len(X)
    # Each cluster is a list of point indices
    clusters = {i: [i] for i in range(n)}
    
    def cluster_dist(a, b):
        pts_a = np.array([X[i] for i in clusters[a]])
        pts_b = np.array([X[i] for i in clusters[b]])
        # pairwise distances
        diffs = pts_a[:, None, :] - pts_b[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        if linkage == 'single': return dists.min()
        if linkage == 'complete': return dists.max()
        return dists.mean()  # wrong for average: should be sum/|A||B|
    
    while len(clusters) > n_clusters:
        ids = sorted(clusters.keys())
        best = None
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                d = cluster_dist(a, b)
                # Wrong tiebreak: just (d, a) instead of (d, min(a,b), max(a,b))
                if best is None or d < best[0]:
                    best = (d, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    
    final = sorted(clusters.keys())
    labels = [0] * n
    for label, cid in enumerate(final):
        for idx in clusters[cid]:
            labels[idx] = label
    return labels
