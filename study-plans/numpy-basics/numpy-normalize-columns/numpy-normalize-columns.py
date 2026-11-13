import numpy as np

def normalize(data: list) -> np.ndarray:  
    a = np.array(data, dtype=np.float64)
    means = a.mean(axis=0)
    stds = a.std(axis=0, ddof=0)
    return (a - means) / stds
