
import numpy as np

def rbf_kernel_matrix(X: list, gamma: float) -> np.ndarray:
    n = len(X)
    K = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            diff = np.array(X[i]) - np.array(X[j]) 
            K[i,j] = np.exp(-gamma * np.dot(diff, diff))

    return K
