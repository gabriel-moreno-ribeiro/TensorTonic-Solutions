import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a float.
    """
    a_array = np.asarray(a, dtype=np.float64)
    b_array = np.asarray(b, dtype=np.float64)
    denominator = np.linalg.norm(a_array) * np.linalg.norm(b_array)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a_array, b_array) / denominator)