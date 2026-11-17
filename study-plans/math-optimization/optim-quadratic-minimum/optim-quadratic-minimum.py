def quadratic_minimum(a: float, b: float, c: float) -> dict:
    x_star = -b / (2 * a)
    f_min = c - (b ** 2) / (4 * a)
    return {"x_star": round(x_star, 6), "f_min": round(f_min, 6)}