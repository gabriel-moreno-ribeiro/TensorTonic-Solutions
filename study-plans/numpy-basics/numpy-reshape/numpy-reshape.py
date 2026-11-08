import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    arr = np.array(data, dtype=np.float64)
    if operation == "flatten":
        return arr.flatten()
    elif operation == "transpose":
        return arr.T
    else:
        return arr.reshape(1, arr.shape[0], arr.shape[1])
