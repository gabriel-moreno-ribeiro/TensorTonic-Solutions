import numpy as np

def gradient_clipping(X: list, y: list, lr: float, clip_value: float, clip_norm: float, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    
    # Three independent runs
    w1 = np.zeros(d); b1 = 0.0  # unclipped
    w2 = np.zeros(d); b2 = 0.0  # value clipped
    w3 = np.zeros(d); b3 = 0.0  # norm clipped
    
    unclipped_losses = []
    clip_value_losses = []
    clip_norm_losses = []
    
    for ep in range(n_epochs):
        # Unclipped
        r1 = X @ w1 + b1 - y
        gw1 = (2.0/n) * X.T @ r1
        gb1 = (2.0/n) * np.sum(r1)
        w1 -= lr * gw1; b1 -= lr * gb1
        unclipped_losses.append(float(np.mean((X @ w1 + b1 - y)**2)))
        
        # Value clipping
        r2 = X @ w2 + b2 - y
        gw2 = (2.0/n) * X.T @ r2
        gb2 = (2.0/n) * np.sum(r2)
        gw2 = np.clip(gw2, -clip_value, clip_value)
        gb2 = float(np.clip(gb2, -clip_value, clip_value))
        w2 -= lr * gw2; b2 -= lr * gb2
        clip_value_losses.append(float(np.mean((X @ w2 + b2 - y)**2)))
        
        # Norm clipping
        r3 = X @ w3 + b3 - y
        gw3 = (2.0/n) * X.T @ r3
        gb3 = (2.0/n) * np.sum(r3)
        g_all = np.concatenate([gw3, [gb3]])
        norm = np.linalg.norm(g_all)
        if norm > clip_norm:
            scale = clip_norm / norm
            gw3 = gw3 * scale
            gb3 = gb3 * scale
        w3 -= lr * gw3; b3 -= lr * float(gb3)
        clip_norm_losses.append(float(np.mean((X @ w3 + b3 - y)**2)))
    
    return {
        "unclipped_losses": unclipped_losses,
        "clip_value_losses": clip_value_losses,
        "clip_norm_losses": clip_norm_losses
    }
