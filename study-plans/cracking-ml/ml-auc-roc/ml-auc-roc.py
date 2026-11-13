import numpy as np
from itertools import groupby

def auc_roc(y_true: list, y_scores: list) -> tuple:
    y = np.array(y_true)
    s = np.array(y_scores, dtype=np.float64)
    
    P = int(y.sum())
    N = len(y) - P
    
    # Stable sort descending by score
    order = np.argsort(-s, kind='stable')
    y_s = y[order].tolist()
    s_s = s[order].tolist()
    
    fprs = [0.0]
    tprs = [0.0]
    tp = fp = 0
    
    i = 0
    while i < len(s_s):
        cur_score = s_s[i]
        # collect all with same score
        while i < len(s_s) and s_s[i] == cur_score:
            if y_s[i] == 1: tp += 1
            else: fp += 1
            i += 1
        fprs.append(round(fp / N, 4) if N > 0 else 0.0)
        tprs.append(round(tp / P, 4) if P > 0 else 0.0)
    
    auc = sum(
        (fprs[k] - fprs[k-1]) * (tprs[k] + tprs[k-1]) / 2
        for k in range(1, len(fprs))
    )
    
    return (fprs, tprs, round(float(auc), 4))
