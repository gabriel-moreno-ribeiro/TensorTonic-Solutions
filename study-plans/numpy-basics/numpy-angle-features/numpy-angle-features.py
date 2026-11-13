import numpy as np

def angle_features(angles: list) -> np.ndarray:
    a = np.array(angles, dtype=np.float64)
    return np.stack([np.sin(a), np.cos(a), np.tan(a)])
