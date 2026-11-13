
import numpy as np

def lda_classify(X_train: list, y_train: list, X_test: list) -> list:
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)
    
    classes = np.unique(y_train)
    n, d = X_train.shape
    K = len(classes)
    
    means = {}
    priors = {}
    for c in classes:
        mask = y_train == c
        means[c] = X_train[mask].mean(axis=0)
        priors[c] = mask.sum() / n
    
    S_w = np.zeros((d, d))
    for c in classes:
        mask = y_train == c
        Xc = X_train[mask] - means[c]
        S_w += Xc.T @ Xc
    Sigma = S_w / (n - K) + 1e-6 * np.eye(d)
    
    preds = []
    for x in X_test:
        scores = {}
        for c in classes:
            mu = means[c]
            # solve Sigma @ v = mu, then score = x.T @ v - 0.5 * mu.T @ v + log(prior)
            v = np.linalg.solve(Sigma, mu)
            scores[c] = x @ v - 0.5 * mu @ v + np.log(priors[c])
        best = min(classes, key=lambda c: (-scores[c], c))
        preds.append(int(best))
    return preds
