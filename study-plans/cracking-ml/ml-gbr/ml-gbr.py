import numpy as np

def mse(y):
    if len(y) == 0: return 0.0
    return float(np.mean((y - np.mean(y))**2))

def best_split(X, y):
    n, d = X.shape; best_gain = 0.0; best_feat = best_thresh = None
    m = mse(y)
    for j in range(d):
        for t in np.unique(X[:, j]):
            lmask = X[:, j] <= t
            left = y[lmask]; right = y[~lmask]
            if len(left) == 0 or len(right) == 0: continue
            gain = m - (len(left)/n * mse(left) + len(right)/n * mse(right))
            if gain > best_gain:
                best_gain = gain; best_feat = j; best_thresh = float(t)
    return best_feat, best_thresh

def build_tree(X, y, depth, max_depth):
    if depth >= max_depth or len(y) == 0:
        return float(np.mean(y)) if len(y) > 0 else 0.0
    feat, thresh = best_split(X, y)
    if feat is None: return float(np.mean(y))
    lmask = X[:, feat] <= thresh
    return {"f": feat, "t": thresh,
            "l": build_tree(X[lmask], y[lmask], depth+1, max_depth),
            "r": build_tree(X[~lmask], y[~lmask], depth+1, max_depth)}

def predict_one(node, x):
    if not isinstance(node, dict): return node
    return predict_one(node["l"], x) if x[node["f"]] <= node["t"] else predict_one(node["r"], x)

def gbr_predict(X_train, y_train, X_test, n_estimators=10, max_depth=3, learning_rate=0.1, seed=42):
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    F = np.full(len(y_train), np.mean(y_train))
    F0 = np.mean(y_train)
    for _ in range(n_estimators):
        r = y_train - F
        tree = build_tree(X_train, r, 0, max_depth)
        F += learning_rate * np.array([predict_one(tree, x) for x in X_train])
    F_test = np.full(len(X_test), F0)
    # rebuild trees
    F2 = np.full(len(y_train), np.mean(y_train))
    trees2 = []
    for _ in range(n_estimators):
        r2 = y_train - F2
        tree2 = build_tree(X_train, r2, 0, max_depth)
        trees2.append(tree2)
        F2 += learning_rate * np.array([predict_one(tree2, x) for x in X_train])
    preds = np.full(len(X_test), np.mean(y_train))
    for tree2 in trees2:
        preds += learning_rate * np.array([predict_one(tree2, x) for x in X_test])
    return [round(float(p), 4) for p in preds]
