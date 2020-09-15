from numpy.random import rand
from cmdtools import utils
from cmdtools.estimation import galerkin


def test_propagator():
    x = rand(10, 3)
    centers = rand(4, 3)
    traj = galerkin.Gaussian(timeseries=x, centers=centers)
    P = traj.propagator
    assert P.shape == (4, 4)
    assert utils.is_rowstochastic(P)

# TODO: need tests checking if the results make any sense


def test_gaussian_bench(benchmark):
    x = rand(1000, 2)
    c = rand(100, 2)
    benchmark(galerkin.Gaussian, x, c)
