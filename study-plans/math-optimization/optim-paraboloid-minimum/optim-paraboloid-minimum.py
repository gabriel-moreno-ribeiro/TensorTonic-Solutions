def paraboloid_minimum(a: float, b: float, c: float, d: float, e: float) -> dict:
    x_star = -c / (2 * a)
    y_star = -d / (2 * b)
    f_min = e - (c**2)/(4*a) - (d**2)/(4*b)
    return {"x_star": round(x_star,6), "y_star": round(y_star,6), "f_min": round(f_min,6)}