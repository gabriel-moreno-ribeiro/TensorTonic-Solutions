import numpy as np

def precision_recall_ap(y_true: list, y_scores: list) -> tuple:
    y = np.array(y_true)
    s = np.array(y_scores, dtype=np.float64)
    P = int(y.sum())
    
    order = np.argsort(-s, kind='stable')
    y_s = y[order].tolist()
    s_s = s[order].tolist()
    
    precs = [1.0]
    recs = [0.0]
    tp = fp = 0
    
    i = 0
    while i < len(s_s):
        cur = s_s[i]
        while i < len(s_s) and s_s[i] == cur:
            if y_s[i] == 1: tp += 1
            else: fp += 1
            i += 1
        precs.append(round(tp / (tp + fp), 4) if (tp + fp) > 0 else 1.0)
        recs.append(round(tp / P, 4) if P > 0 else 0.0)
    
    ap = sum((recs[k] - recs[k-1]) * precs[k] for k in range(1, len(recs)))
    return (precs, recs, round(float(ap), 4))
