import numpy as np

def precision_recall_ap(y_true: list, y_scores: list) -> tuple:
    y = list(map(int, y_true))
    s = list(map(float, y_scores))
    P = sum(y)
    n = len(y)
    
    order = sorted(range(n), key=lambda i: -s[i])
    y_s = [y[i] for i in order]
    s_s = [s[i] for i in order]
    
    precs = [1.0]
    recs = [0.0]
    tp = fp = 0
    
    i = 0
    while i < n:
        cur = s_s[i]
        while i < n and s_s[i] == cur:
            if y_s[i] == 1: tp += 1
            else: fp += 1
            i += 1
        prec = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0
        rec = round(tp / P, 4) if P > 0 else 0.0
        precs.append(prec)
        recs.append(rec)
    
    ap = sum((recs[k] - recs[k-1]) * precs[k] for k in range(1, len(recs)))
    return (precs, recs, round(float(ap), 4))
