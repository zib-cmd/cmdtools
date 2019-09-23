from numpy.random import rand
from cmdtools import utils
from cmdtools.estimation import galerkin

def test_propagator():
    x = rand(10,3)
    centers = rand(4,3)
    sigma = 1
    P = galerkin.propagator(x, centers, sigma)
    assert utils.is_rowstochastic(P)