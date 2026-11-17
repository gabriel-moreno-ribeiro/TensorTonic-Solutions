import numpy as np

def newtons_method(X: list, y: list, lr_gd: float, n_iters: int) -> dict:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n = len(X)
    # Append bias column
    Xa = np.hstack([X, np.ones((n, 1))])
    
    w_gd = np.zeros(Xa.shape[1])
    w_nt = np.zeros(Xa.shape[1])
    
    # Record initial MSE
    gd_losses = [float(np.mean((Xa @ w_gd - y)**2))]
    newton_losses = [float(np.mean((Xa @ w_nt - y)**2))]
    
    # Hessian is constant for linear regression
    H = (2.0/n) * Xa.T @ Xa
    
    for i in range(n_iters):
        # GD step
        g_gd = (2.0/n) * Xa.T @ (Xa @ w_gd - y)
        w_gd = w_gd - lr_gd * g_gd
        gd_losses.append(float(np.mean((Xa @ w_gd - y)**2)))
        
        # Newton step: solve H * delta = g
        g_nt = (2.0/n) * Xa.T @ (Xa @ w_nt - y)
        delta = np.linalg.solve(H, g_nt)
        w_nt = w_nt - delta
        newton_losses.append(float(np.mean((Xa @ w_nt - y)**2)))
    
    return {"gd_losses": gd_losses, "newton_losses": newton_losses}
