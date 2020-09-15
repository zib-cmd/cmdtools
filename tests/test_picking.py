from cmdtools.estimation.picking_algorithm import picking_algorithm
import scipy.spatial.distance as dist
import numpy as np


def test_picking():
    n = 10
    m = 5
    X = np.repeat(np.array([range(n)]).T, axis=1, repeats=2)
    P, inds, d = picking_algorithm(X, m)
    assert np.allclose(P, X[inds, :])
    assert np.allclose(d, dist.cdist(X, P, metric='sqeuclidean'))


def test_picking_bench(benchmark):
    x = np.random.rand(1000, 2)
    benchmark(picking_algorithm, x, 100)
