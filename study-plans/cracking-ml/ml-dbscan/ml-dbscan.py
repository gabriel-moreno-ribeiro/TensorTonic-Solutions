import numpy as np

def dbscan(X: list, eps: float = 0.5, min_samples: int = 5) -> list:
    X = np.array(X, dtype=np.float64)
    n = len(X)
    labels = [-1] * n
    
    def neighbors(i):
        diffs = X - X[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        return sorted(np.where(dists <= eps)[0].tolist())
    
    cluster = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        nbrs = neighbors(i)
        if len(nbrs) < min_samples:
            continue  # noise for now
        # core point: start cluster
        labels[i] = cluster
        queue = [j for j in nbrs if j != i]
        in_queue = set(queue)
        while queue:
            q = queue.pop(0)
            if labels[q] == -1:
                labels[q] = cluster
            elif labels[q] != -1:
                continue
            q_nbrs = neighbors(q)
            if len(q_nbrs) >= min_samples:
                for nb in q_nbrs:
                    if nb not in in_queue and labels[nb] == -1:
                        queue.append(nb)
                        in_queue.add(nb)
        cluster += 1
    
    return labels
