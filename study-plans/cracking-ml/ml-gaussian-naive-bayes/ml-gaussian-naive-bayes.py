
import numpy as np

def gaussian_naive_bayes(X_train: list, y_train: list, X_test: list) -> list:
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)
    
    classes = np.unique(y_train)
    n_total = len(y_train)
    
    priors = {}
    means = {}
    variances = {}
    
    for c in classes:
        mask = y_train == c
        priors[c] = mask.sum() / n_total
        means[c] = X_train[mask].mean(axis=0)
        variances[c] = X_train[mask].var(axis=0, ddof=1) + 1e-9  # bug: ddof=1
    
    preds = []
    for x in X_test:
        best_score = -np.inf
        best_class = None
        for c in classes:
            score = np.log(priors[c]) - 0.5 * np.sum(
                np.log(2 * np.pi * variances[c]) + (x - means[c])**2 / variances[c]
            )
            if score > best_score:
                best_score = score
                best_class = c
        preds.append(int(best_class))
    return preds
