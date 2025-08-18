def Distance(t1, t2):
    return sum((x - y)**2 for x, y in zip(t1, t2))**0.5
