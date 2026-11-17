import numpy as np

def vanilla_gradient_descent(x0: float, y0: float, lr: float, n_iters: int) -> dict:
    x, y = x0, y0
    trajectory = [(x, y)]
    for _ in range(n_iters):
        grad_x = 2 * x
        grad_y = 6 * y
        x = x - lr * grad_x
        y = y - lr * grad_y
        trajectory.append((x, y))
    final_point = (x, y)
    final_value = x**2 + 3*y**2
    return {'trajectory': trajectory, 'final_point': final_point, 'final_value': final_value}
