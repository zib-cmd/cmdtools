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


def test_voronoi_equality():
    # issue #25 suggests that we cannot correctly assign centers to the centers
    # here we check that they get correctly assigned
    n = 100
    X = np.random.rand(n, 5)
    assert np.allclose(voronoi.by_nn(X, X)[1], np.arange(n))


def test_sparseboxes():
    X = np.random.rand(100, 2)
    X[-1,:] = X[0,:]  # fixes warning about non-ergodicity
    t = voronoi.SparseBoxes(X, ns=5)
    t.propagator()
