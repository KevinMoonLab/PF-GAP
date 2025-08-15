import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

# Code for Experiment 4 functions

#& Data Constants
SEEDS = [61, 737, 821, 161, 346, 78, 2, 67, 102, 982]

#& Generate Data Functions
def create_3d_sphere_data(seed):
    """Creates two clusters on a 3d sphere"""

    np.random.seed(seed)
    gs.random.seed(seed)

    sphere = Hypersphere(dim=2, equip=False)
    cluster = sphere.random_von_mises_fisher(kappa=20, n_samples=150)

    SO3 = SpecialOrthogonal(3, equip=False)
    rotation1 = SO3.random_uniform()
    rotation2 = SO3.random_uniform()

    cluster_1 = cluster @ rotation1
    cluster_2 = cluster @ rotation2

    #Create labels
    labels_1 = np.zeros(cluster_1.shape[0])
    labels_2 = np.ones(cluster_2.shape[0])
    labels = np.concatenate([labels_1, labels_2])

    #Combine data
    data = np.concatenate([cluster_1, cluster_2], axis=0)
    
    return data, labels

def to_spherical_coords(data):
    """Convert 3D points on the unit sphere to 2D spherical coordinates (polar, azimuthal)."""
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    r = np.linalg.norm(data, axis=1)
    r = np.where(r == 0, 1e-8, r)
    polar = np.arccos(z / r)
    azimuth = np.arctan2(y, x)
    return np.vstack((polar, azimuth)).T