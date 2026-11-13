import numpy as np

def gmm(X: list, k: int, max_iters: int = 100, seed: int = 42) -> tuple:
    X = np.array(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.RandomState(seed)
    
    # Init: k distinct random rows
    idx = rng.choice(n, k, replace=False)
    means = X[idx].copy()
    variances = np.ones((k, d))
    weights = np.ones(k) / k
    
    for _ in range(max_iters):
        # E-step: log responsibilities (unstabilized - wrong)
        log_resp = np.zeros((n, k))
        for j in range(k):
            diff = X - means[j]
            log_resp[:, j] = (np.log(weights[j]) 
                - 0.5 * np.sum(np.log(2*np.pi*variances[j]) + diff**2/variances[j], axis=1))
        # Softmax without max subtraction
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        
        # M-step
        Nk = resp.sum(axis=0)
        weights = Nk / n
        means = (resp.T @ X) / Nk[:, None]
        variances = np.array([
            (resp[:, j:j+1] * (X - means[j])**2).sum(axis=0) / Nk[j]
            for j in range(k)
        ]) + 1e-6
    
    # Final E-step
    log_resp = np.zeros((n, k))
    for j in range(k):
        diff = X - means[j]
        log_resp[:, j] = (np.log(weights[j]) 
            - 0.5 * np.sum(np.log(2*np.pi*variances[j]) + diff**2/variances[j], axis=1))
    labels = np.argmax(log_resp, axis=1).tolist()
    
    return (labels, 
            [[round(float(v), 4) for v in m] for m in means],
            [round(float(w), 4) for w in weights])
