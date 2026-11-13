import numpy as np
from collections import Counter

def majority(y):
    c = Counter(int(v) for v in y)
    return min(c, key=lambda k: (-c[k], k))

def gini(y):
    if len(y) == 0: return 0.0
    n = len(y); counts = Counter(int(v) for v in y)
    return 1.0 - sum((c/n)**2 for c in counts.values())

def stump_predict(X_train, y_train, X_pred):
    n, d = X_train.shape
    best_gain = 0.0; best_j = best_t = None
    g = gini(y_train)
    for j in range(d):
        for t in np.unique(X_train[:, j]):
            lmask = X_train[:, j] <= t
            left = y_train[lmask]; right = y_train[~lmask]
            if len(left) == 0 or len(right) == 0: continue
            gain = g - (len(left)/n * gini(left) + len(right)/n * gini(right))
            if round(gain, 6) > round(best_gain, 6):
                best_gain = gain; best_j = j; best_t = float(t)
    if best_j is None:
        label = majority(y_train)
        return np.array([label]*len(X_pred))
    left_label = majority(y_train[X_train[:, best_j] <= best_t])
    right_label = majority(y_train[X_train[:, best_j] > best_t])
    return np.where(X_pred[:, best_j] <= best_t, left_label, right_label)

def knn3_predict(X_train, y_train, X_pred):
    preds = []
    for x in X_pred:
        dists = np.sum((X_train - x)**2, axis=1)
        idx = np.argsort(dists, kind='stable')[:3]
        preds.append(majority(y_train[idx]))
    return np.array(preds)

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def stacking_classify(X_train, y_train, X_test, n_folds=3, seed=42):
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float64)
    n = len(y_train); classes = sorted(set(y_train.tolist()))
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    # Out-of-fold predictions
    oof = np.zeros((n, 2), dtype=np.float64)
    for fi in range(n_folds):
        val_idx = folds[fi]
        tr_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fi])
        Xtr, ytr = X_train[tr_idx], y_train[tr_idx]
        Xval = X_train[val_idx]
        oof[val_idx, 0] = stump_predict(Xtr, ytr, Xval)
        oof[val_idx, 1] = knn3_predict(Xtr, ytr, Xval)
    # Test predictions from full models
    stump_test = stump_predict(X_train, y_train, X_test)
    knn_test = knn3_predict(X_train, y_train, X_test)
    test_meta = np.column_stack([stump_test, knn_test]).astype(np.float64)
    # One-vs-rest logistic meta-model
    n_classes = len(classes)
    scores = np.zeros((len(X_test), n_classes))
    for ci, c in enumerate(classes):
        binary = (y_train == c).astype(np.float64)
        w = np.zeros(2); b = 0.0
        for _ in range(500):
            p = sigmoid(oof @ w + b)
            err = p - binary
            w -= 0.1 * oof.T @ err / n
            b -= 0.1 * np.sum(err) / n
        scores[:, ci] = test_meta @ w + b
    return [int(classes[np.argmax(scores[i])]) for i in range(len(X_test))]
