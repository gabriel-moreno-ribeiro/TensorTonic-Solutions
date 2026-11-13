import numpy as np
from collections import Counter

def gini(y):
    if len(y) == 0: return 0.0
    n = len(y); counts = Counter(int(v) for v in y)
    return 1.0 - sum((c/n)**2 for c in counts.values())

def majority(y):
    c = Counter(int(v) for v in y)
    return min(c, key=lambda k: (-c[k], k))

def n_features(d, max_features):
    if max_features == 'sqrt': return max(1, int(np.floor(np.sqrt(d))))
    if max_features == 'log2': return max(1, int(np.floor(np.log2(d))))
    return min(d, int(max_features))

def best_split(X, y, rng, max_features):
    n, d = X.shape; best_gain = 0.0; best_feat = best_thresh = None
    m = n_features(d, max_features)
    feats = sorted(rng.choice(d, size=m, replace=False).tolist())
    g = gini(y)
    for j in feats:
        for t in np.unique(X[:, j]):
            lmask = X[:, j] <= t
            left = y[lmask]; right = y[~lmask]
            if len(left) == 0 or len(right) == 0: continue
            gain = g - (len(left)/n * gini(left) + len(right)/n * gini(right))
            if gain > best_gain:  # no round - wrong
                best_gain = gain; best_feat = j; best_thresh = float(t)
    return best_feat, best_thresh

def build_tree(X, y, depth, max_depth, rng, max_features):
    if depth >= max_depth or len(set(int(v) for v in y)) == 1:
        return majority(y)
    feat, thresh = best_split(X, y, rng, max_features)
    if feat is None: return majority(y)
    lmask = X[:, feat] <= thresh
    return {"f": feat, "t": thresh,
            "l": build_tree(X[lmask], y[lmask], depth+1, max_depth, rng, max_features),
            "r": build_tree(X[~lmask], y[~lmask], depth+1, max_depth, rng, max_features)}

def predict_one(node, x):
    if not isinstance(node, dict): return node
    return predict_one(node["l"], x) if x[node["f"]] <= node["t"] else predict_one(node["r"], x)

def random_forest_classify(X_train, y_train, X_test, n_estimators=10, max_depth=5, max_features="sqrt", seed=42):
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)
    n = len(y_train); rng = np.random.RandomState(seed)
    trees = []
    for _ in range(n_estimators):
        idx = rng.randint(0, n, size=n)
        tree = build_tree(X_train[idx], y_train[idx], 0, max_depth, rng, max_features)
        trees.append(tree)
    preds = []
    for x in X_test:
        votes = [predict_one(t, x) for t in trees]
        preds.append(int(majority(np.array(votes))))
    return preds
