def early_stopping(val_losses: list, patience: int) -> dict:
    best_epoch = 0
    best_loss = val_losses[0]
    wait = 0
    stop_epoch = len(val_losses) - 1
    
    for i in range(1, len(val_losses)):
        if val_losses[i] < best_loss:
            best_loss = val_losses[i]
            best_epoch = i
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                stop_epoch = i
                return {"best_epoch": best_epoch, "stop_epoch": stop_epoch}
    
    return {"best_epoch": best_epoch, "stop_epoch": stop_epoch}
