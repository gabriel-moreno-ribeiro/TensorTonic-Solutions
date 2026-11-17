import numpy as np

def optimizer_showdown(X: list, y: list, lr: float, beta1: float, beta2: float, epsilon: float, n_epochs: int, batch_size: int) -> dict:
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    n, d = X.shape
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def bce(w, b):
        p = sigmoid(X @ w + b)
        return float(-np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15)))
    
    # SGD
    w_sgd = np.zeros(d); b_sgd = 0.0
    # Momentum
    w_mom = np.zeros(d); b_mom = 0.0
    vw_mom = np.zeros(d); vb_mom = 0.0
    # Adam
    w_adam = np.zeros(d); b_adam = 0.0
    mw = np.zeros(d); mb = 0.0
    vw = np.zeros(d); vb = 0.0
    t_adam = 0  # timestep for Adam
    
    sgd_losses = []
    momentum_losses = []
    adam_losses = []
    
    for epoch in range(n_epochs):
        for start in range(0, n, batch_size):
            Xb = X[start:start+batch_size]
            yb = y[start:start+batch_size]
            r = len(Xb)
            
            # SGD gradients
            p_sgd = sigmoid(Xb @ w_sgd + b_sgd)
            gw_sgd = (1.0/r) * Xb.T @ (p_sgd - yb)
            gb_sgd = (1.0/r) * np.sum(p_sgd - yb)
            w_sgd -= lr * gw_sgd
            b_sgd -= lr * gb_sgd
            
            # Momentum gradients
            p_mom = sigmoid(Xb @ w_mom + b_mom)
            gw_mom = (1.0/r) * Xb.T @ (p_mom - yb)
            gb_mom = (1.0/r) * np.sum(p_mom - yb)
            vw_mom = beta1 * vw_mom + gw_mom
            vb_mom = beta1 * vb_mom + gb_mom
            w_mom -= lr * vw_mom
            b_mom -= lr * vb_mom
            
            # Adam gradients
            p_adam = sigmoid(Xb @ w_adam + b_adam)
            gw_adam = (1.0/r) * Xb.T @ (p_adam - yb)
            gb_adam = (1.0/r) * np.sum(p_adam - yb)
            t_adam += 1
            mw = beta1 * mw + (1 - beta1) * gw_adam
            mb = beta1 * mb + (1 - beta1) * gb_adam
            vw = beta2 * vw + (1 - beta2) * gw_adam**2
            vb = beta2 * vb + (1 - beta2) * gb_adam**2
            mw_hat = mw / (1 - beta1**t_adam)
            mb_hat = mb / (1 - beta1**t_adam)
            vw_hat = vw / (1 - beta2**t_adam)
            vb_hat = vb / (1 - beta2**t_adam)
            w_adam -= lr * mw_hat / (np.sqrt(vw_hat) + epsilon)
            b_adam -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)
        
        sgd_losses.append(bce(w_sgd, b_sgd))
        momentum_losses.append(bce(w_mom, b_mom))
        adam_losses.append(bce(w_adam, b_adam))
    
    return {"sgd_losses": sgd_losses, "momentum_losses": momentum_losses, "adam_losses": adam_losses}
