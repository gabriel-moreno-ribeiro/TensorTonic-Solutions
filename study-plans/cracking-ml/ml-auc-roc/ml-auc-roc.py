import numpy as np

def auc_roc(y_true: list, y_scores: list) -> tuple:
    y = np.array(y_true)
    s = np.array(y_scores, dtype=np.float64)
    
    P = int(y.sum())
    N = len(y) - P
    
    # Stable sort descending
    order = np.argsort(-s, kind='stable')
    y_sorted = y[order]
    s_sorted = s[order]
    
    fprs = [0.0]
    tprs = [0.0]
    tp = fp = 0
    
    i = 0
    n = len(y_sorted)
    while i < n:
        # Find group of equal scores
        j = i
        while j < n and s_sorted[j] == s_sorted[i]:
            j += 1
        # Process group [i, j)
        for k in range(i, j):
            if y_sorted[k] == 1: tp += 1
            else: fp += 1
        # Append one point after group
        fprs.append(round(float(fp / N), 4) if N > 0 else 0.0)
        tprs.append(round(float(tp / P), 4) if P > 0 else 0.0)
        i = j
    
    # AUC via trapezoid
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i-1]) * (tprs[i] + tprs[i-1]) / 2
    
    return (fprs, tprs, round(float(auc), 4))
