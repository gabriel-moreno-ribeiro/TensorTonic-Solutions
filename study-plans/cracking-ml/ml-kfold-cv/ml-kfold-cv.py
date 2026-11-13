import numpy as np

def kfold_cv(X: list, y: list, model_fn, k: int = 5, seed: int = 42) -> tuple:
    X = np.array(X, dtype=np.float64)
    y = np.array(y)
    n = len(y)
    
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    
    accs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        pred_fn = model_fn(X_train, y_train)
        preds = np.array(pred_fn(X_val))
        correct = np.sum(preds == y_val)
        acc = round(float(correct / len(y_val)), 4)
        accs.append(acc)
    
    mean_acc = round(float(sum(accs) / k), 4)
    return (accs, mean_acc)
