import numpy as np
from collections import Counter

def gini(y):
    if len(y) == 0: return 0.0
    n = len(y)
    counts = Counter(int(v) for v in y)
    return 1.0 - sum((c/n)**2 for c in counts.values())

def majority(y):
    c = Counter(int(v) for v in y)
    return min(c, key=lambda k: (-c[k], k))

def best_split(X, y):
    n, d = X.shape; best_gain = 0.0; best_feat = best_thresh = None
    g = gini(y)
    for j in range(d):
        thresholds = np.unique(X[:, j])
        for t in thresholds:
            lmask = X[:, j] <= t
            left = y[lmask]; right = y[~lmask]
            if len(left) == 0 or len(right) == 0: continue
            gain = g - (len(left)/n * gini(left) + len(right)/n * gini(right))
            if round(gain, 10) > round(best_gain, 10):
                best_gain = gain; best_feat = j; best_thresh = float(t)
    return best_feat, best_thresh

def build_tree(X, y, depth, max_depth, min_samples):
    if depth >= max_depth or len(y) < min_samples or len(set(int(v) for v in y)) == 1:
        return majority(y)
    feat, thresh = best_split(X, y)
    if feat is None:
        return majority(y)
    lmask = X[:, feat] <= thresh
    return {"f": feat, "t": thresh,
            "l": build_tree(X[lmask], y[lmask], depth+1, max_depth, min_samples),
            "r": build_tree(X[~lmask], y[~lmask], depth+1, max_depth, min_samples)}

def predict_one(node, x):
    if not isinstance(node, dict): return node
    return predict_one(node["l"], x) if x[node["f"]] <= node["t"] else predict_one(node["r"], x)

def cart_classify(X_train, y_train, X_test, max_depth=5, min_samples=2):
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)
    tree = build_tree(X_train, y_train, 0, max_depth, min_samples)
    return [int(predict_one(tree, x)) for x in X_test]
