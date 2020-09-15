import cmdtools.estimation.voronoi as voronoi
import numpy as np


def test_voronoi():
    X = np.random.rand(100, 5)
    for method in ['kmeans', 'picking']:
        v = voronoi.VoronoiTrajectory(X, 5, centers=method)
        v.propagator()


def test_voronoi_nn():
    X = np.random.rand(100, 5)
    centers = np.random.rand(10, 5)
    voronoi.VoronoiTrajectory(X, 5, centers=centers)
