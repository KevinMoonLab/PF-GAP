def Distance(t1, t2):
    from geomstats.geometry.hypersphere import Hypersphere
    sphere = Hypersphere(dim=2)
    return sphere.metric.dist(t1, t2)
