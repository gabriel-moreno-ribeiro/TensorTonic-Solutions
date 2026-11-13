
import numpy as np
from collections import Counter

def knn_classifier(X_train: list, y_train: list, X_test: list, k: int) -> list:
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=np.float64)
    preds = []
    for x in X_test:
        dists = np.sqrt(np.sum((X_train - x)**2, axis=1))
        # stable sort preserves original index order for equal distances
        idx = np.argsort(dists, kind='stable')[:k]
        neighbors = y_train[idx]
        counts = Counter(neighbors.tolist())
        # ties broken by smallest label
        pred = min(counts, key=lambda c: (-counts[c], c))
        preds.append(int(pred))
    return preds
