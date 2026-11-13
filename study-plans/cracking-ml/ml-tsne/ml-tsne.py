import numpy as np

def tsne_affinity(X: list, perplexity: float = 5.0, tol: float = 1e-5, max_binary_search: int = 50) -> list:
    X = np.array(X, dtype=np.float64)
    n = len(X)
    target = np.log(perplexity)
    
    sq_dists = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    
    cond_P = np.zeros((n, n))
    
    for i in range(n):
        d = sq_dists[i].copy()
        
        beta = 1.0
        beta_lo = -np.inf
        beta_hi = np.inf
        
        p_i = np.zeros(n)
        for _ in range(max_binary_search):
            neg_bd = -beta * d
            neg_bd[i] = -np.inf  # exclude self
            # log-sum-exp stable
            mx = neg_bd[np.isfinite(neg_bd)].max()
            exp_d = np.exp(neg_bd - mx)
            exp_d[i] = 0.0
            sum_exp = exp_d.sum()
            p_i = exp_d / sum_exp if sum_exp > 0 else np.zeros(n)
            
            nz = p_i > 0
            H = -np.sum(p_i[nz] * np.log(p_i[nz]))
            
            diff = H - target
            if abs(diff) < tol:
                break
            
            if diff > 0:
                beta_lo = beta
                beta = beta * 2 if np.isinf(beta_hi) else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = beta / 2 if np.isinf(beta_lo) else (beta + beta_lo) / 2
        
        cond_P[i] = p_i
    
    P = (cond_P + cond_P.T) / (2 * n)
    return [[round(float(v), 4) for v in row] for row in P]
