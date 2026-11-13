def categorical_encode(data: list, method: str = "label"):
    classes = sorted(set(data))
    idx = {c: i for i, c in enumerate(classes)}
    if method == "label":
        return {"classes": classes, "encoded": [idx[x] for x in data]}
    else:
        n = len(classes)
        return [[1 if j == idx[x] else 0 for j in range(n)] for x in data]
