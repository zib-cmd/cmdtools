from numpy.random import rand
from cmdtools import utils
from cmdtools.estimation import galerkin, find_bandwidth


def test_propagator():
    x = rand(10, 3)
    centers = rand(4, 3)
    sigma = find_bandwidth(X, centers)
    P = galerkin.propagator(x, centers, sigma)
    assert utils.is_rowstochastic(P)
