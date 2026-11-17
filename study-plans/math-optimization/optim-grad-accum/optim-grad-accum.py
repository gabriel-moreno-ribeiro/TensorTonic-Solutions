import numpy as np

def gradient_accumulation(X: list, y: list, lr: float, micro_batch: int, accum_steps: int, n_epochs: int) -> dict:
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n, d = X.shape
    
    w_no = np.zeros(d)
    w_ac = np.zeros(d)
    no_accum_losses = []
    accum_losses = []
    
    for epoch in range(n_epochs):
        # No accumulation: update after every micro-batch
        for start in range(0, n, micro_batch):
            Xb = X[start:start+micro_batch]
            yb = y[start:start+micro_batch]
            B = len(Xb)
            g = (2.0/B) * Xb.T @ (Xb @ w_no - yb)
            w_no -= lr * g
        
        # Gradient accumulation: accumulate accum_steps micro-batches, then update once
        micro_starts = list(range(0, n, micro_batch))
        i = 0
        while i < len(micro_starts):
            # collect up to accum_steps micro-batches
            window = micro_starts[i:i+accum_steps]
            G_sum = np.zeros(d)
            total_samples = 0
            for s in window:
                Xb = X[s:s+micro_batch]
                yb = y[s:s+micro_batch]
                B = len(Xb)
                # unnormalized gradient sum for this micro-batch
                G_sum += 2 * Xb.T @ (Xb @ w_ac - yb)
                total_samples += B
            # divide by total accumulated samples, then update
            g_acc = G_sum / total_samples
            w_ac -= lr * g_acc
            i += accum_steps
        
        mse_no = float(np.mean((X @ w_no - y)**2))
        mse_ac = float(np.mean((X @ w_ac - y)**2))
        no_accum_losses.append(mse_no)
        accum_losses.append(mse_ac)
    
    return {
        "no_accum_losses": no_accum_losses,
        "accum_losses": accum_losses,
        "no_accum_weights": [float(x) for x in w_no],
        "accum_weights": [float(x) for x in w_ac]
    }
