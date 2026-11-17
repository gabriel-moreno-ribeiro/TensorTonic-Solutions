import numpy as np

def dropout_mlp(X_train: list, y_train: list, X_test: list, y_test: list, hidden_size: int, lr: float, dropout_rate: float, n_epochs: int, seed: int) -> list:
    np.random.seed(seed)
    X_tr = np.array(X_train, dtype=float)
    y_tr = np.array(y_train, dtype=float).reshape(-1, 1)
    X_te = np.array(X_test, dtype=float)
    y_te = np.array(y_test, dtype=float)
    n, d = X_tr.shape
    
    # He init: scale = sqrt(2 / fan_in)
    W1 = np.random.randn(d, hidden_size) * np.sqrt(2.0 / d)
    b1 = np.zeros(hidden_size)
    W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
    b2 = np.zeros(1)
    
    accs = []
    for ep in range(n_epochs):
        # Forward with inverted dropout
        z1 = X_tr @ W1 + b1
        a1 = np.maximum(0, z1)  # ReLU
        
        # Dropout mask: keep if rand > dropout_rate
        mask = (np.random.rand(*a1.shape) > dropout_rate).astype(float)
        if dropout_rate < 1.0:
            a1_drop = a1 * mask / (1.0 - dropout_rate)
        else:
            a1_drop = np.zeros_like(a1)
        
        z2 = a1_drop @ W2 + b2
        out = 1.0 / (1.0 + np.exp(-z2))
        
        # Backprop
        d_out = (out - y_tr) / n
        d_W2 = a1_drop.T @ d_out
        d_b2 = np.sum(d_out, axis=0)
        d_a1_drop = d_out @ W2.T
        # Apply same mask in backprop
        d_a1 = d_a1_drop * mask / (1.0 - dropout_rate) if dropout_rate < 1.0 else np.zeros_like(d_a1_drop)
        d_z1 = d_a1 * (z1 > 0)
        d_W1 = X_tr.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0)
        
        W1 -= lr * d_W1
        b1 -= lr * d_b1
        W2 -= lr * d_W2
        b2 -= lr * d_b2
        
        # Test accuracy WITHOUT dropout
        z1_te = X_te @ W1 + b1
        a1_te = np.maximum(0, z1_te)
        z2_te = a1_te @ W2 + b2
        preds = (1.0 / (1.0 + np.exp(-z2_te))).flatten() >= 0.5
        acc = float(np.mean(preds == y_te))
        accs.append(acc)
    
    return accs
