
import numpy as np

def scaled_dot_product_attention(Q: list, K: list, V: list) -> np.ndarray:
    Q = np.array(Q, dtype=np.float64)
    K = np.array(K, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    
    d_k = Q.shape[1]
    S = Q @ K.T / np.sqrt(d_k)
    
    S_max = np.max(S, axis=1, keepdims=True)
    exp_S = np.exp(S - S_max)
    P = exp_S / np.sum(exp_S, axis=1, keepdims=True)
    
    return P @ V
