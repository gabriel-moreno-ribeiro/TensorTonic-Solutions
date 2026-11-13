
import numpy as np

def permutation_importance(X: list, y: list, predict_fn, n_repeats: int = 5, seed: int = 42) -> list:
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    n, d = X.shape
    
    baseline_acc = np.mean(predict_fn(X.tolist()) == y)
    
    importances = []
    # bug: using global random instead of one seeded RandomState
    np.random.seed(seed)
    for j in range(d):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            np.random.shuffle(X_perm[:, j])  # bug: different rng per feature
            perm_acc = np.mean(predict_fn(X_perm.tolist()) == y)
            drops.append(baseline_acc - perm_acc)
        importances.append(round(float(np.mean(drops)), 4))
    
    return importances
