import numpy as np

def adaboost_classify(X_train, y_train, X_test, n_estimators=10, seed=42):
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    n, d = X_train.shape
    w = np.ones(n) / n
    stumps = []
    for _ in range(n_estimators):
        best_err = float('inf'); best_j = best_t = best_p = None
        for j in range(d):
            for t in np.unique(X_train[:, j]):
                for p in [1, -1]:
                    # WRONG: polarity 1 predicts +1 when x<=t (should be -1)
                    preds = np.where(X_train[:, j] <= t, p, -p)
                    err = np.sum(w * (preds != y_train))
                    if err < best_err:
                        best_err = err; best_j = j; best_t = t; best_p = p
        eps = np.clip(best_err, 1e-12, 1-1e-12)
        alpha = 0.5 * np.log((1-eps)/eps)
        preds = np.where(X_train[:, best_j] <= best_t, best_p, -best_p)
        w *= np.exp(-alpha * y_train * preds)
        w /= w.sum()
        stumps.append((best_j, best_t, best_p, alpha))
    preds = []
    for x in X_test:
        score = sum(alpha * (1 if x[j] <= t else -1) * p for j, t, p, alpha in stumps)
        preds.append(1 if score >= 0 else -1)
    return preds
