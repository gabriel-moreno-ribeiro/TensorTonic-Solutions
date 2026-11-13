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
        pred_fn = model_fn(X[train_idx], y[train_idx])
        preds = pred_fn(X[val_idx])
        acc = round(float(np.mean(np.array(preds) == y[val_idx])), 4)
        accs.append(acc)
    
    mean_acc = round(float(np.mean(accs)), 4)
    return (accs, mean_acc)
