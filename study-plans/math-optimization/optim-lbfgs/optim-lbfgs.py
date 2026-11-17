import numpy as np
from collections import deque

def lbfgs(X: list, y: list, lr_gd: float, m: int, n_iters: int) -> dict:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    
    def mse(w):
        return float(np.mean((X @ w - y)**2))
    
    def grad(w):
        return (2.0/n) * X.T @ (X @ w - y)
    
    # Gradient descent
    w_gd = np.zeros(d)
    gd_losses = [mse(w_gd)]
    for _ in range(n_iters):
        g = grad(w_gd)
        w_gd = w_gd - lr_gd * g
        gd_losses.append(mse(w_gd))
    
    # L-BFGS with two-loop recursion and Armijo backtracking
    w_lbfgs = np.zeros(d)
    lbfgs_losses = [mse(w_lbfgs)]
    s_hist = deque()  # step vectors
    z_hist = deque()  # gradient differences
    
    for iteration in range(n_iters):
        g = grad(w_lbfgs)
        
        # Two-loop recursion
        q = g.copy()
        alphas = []
        pairs = list(zip(s_hist, z_hist))
        
        # First loop (reverse)
        for s, z in reversed(pairs):
            rho = 1.0 / (z.T @ s + 1e-10)
            alpha = rho * (s.T @ q)
            q = q - alpha * z
            alphas.append(alpha)
        
        # Initial Hessian scaling
        if pairs:
            s_last, z_last = pairs[-1]
            gamma = (s_last.T @ z_last) / (z_last.T @ z_last + 1e-10)
            r = gamma * q
        else:
            r = q.copy()  # identity scale = 1
        
        # Second loop (forward)
        for (s, z), alpha in zip(pairs, reversed(alphas)):
            rho = 1.0 / (z.T @ s + 1e-10)
            beta = rho * (z.T @ r)
            r = r + s * (alpha - beta)
        
        direction = -r  # search direction
        
        # Armijo backtracking line search
        f0 = mse(w_lbfgs)
        c = 0.0001
        step = 1.0
        for _ in range(20):
            w_new = w_lbfgs + step * direction
            if mse(w_new) <= f0 + c * step * (g @ direction):
                break
            step *= 0.5
        
        w_new = w_lbfgs + step * direction
        s_new = w_new - w_lbfgs
        z_new = grad(w_new) - g
        
        w_lbfgs = w_new
        
        # Update history
        s_hist.append(s_new)
        z_hist.append(z_new)
        if len(s_hist) > m:
            s_hist.popleft()
            z_hist.popleft()
        
        lbfgs_losses.append(mse(w_lbfgs))
    
    return {"gd_losses": gd_losses, "lbfgs_losses": lbfgs_losses}
