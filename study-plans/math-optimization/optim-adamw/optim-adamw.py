import numpy as np

def adamw_compare(X: list, y: list, lr: float, beta1: float, beta2: float, weight_decay: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    w_l2 = np.zeros(d)
    w_aw = np.zeros(d)
    m_l2, v_l2 = np.zeros(d), np.zeros(d)
    m_aw, v_aw = np.zeros(d), np.zeros(d)
    adam_l2_losses, adamw_losses = [], []
    for t in range(1, n_epochs+1):
        adam_l2_losses.append(float(np.mean((X @ w_l2 - y)**2)))
        g_l2 = (2.0/n) * X.T @ (X @ w_l2 - y) + weight_decay * w_l2
        m_l2 = beta1*m_l2 + (1-beta1)*g_l2
        v_l2 = beta2*v_l2 + (1-beta2)*g_l2**2
        mh = m_l2/(1-beta1**t)
        vh = v_l2/(1-beta2**t)
        w_l2 = w_l2 - lr * mh/(np.sqrt(vh)+1e-8)
        
        adamw_losses.append(float(np.mean((X @ w_aw - y)**2)))
        g_aw = (2.0/n) * X.T @ (X @ w_aw - y)
        m_aw = beta1*m_aw + (1-beta1)*g_aw
        v_aw = beta2*v_aw + (1-beta2)*g_aw**2
        mh_aw = m_aw/(1-beta1**t)
        vh_aw = v_aw/(1-beta2**t)
        w_aw = w_aw * (1 - lr * weight_decay) - lr * mh_aw/(np.sqrt(vh_aw)+1e-8)
    return {'adam_l2_losses': adam_l2_losses, 'adamw_losses': adamw_losses}
