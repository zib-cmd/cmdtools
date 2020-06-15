from cmdtools.estimation.picking_algorithm import picking_algorithm
import scipy.spatial.distance as dist
import numpy as np


def test_picking():
    n = 10
    m = 5
    X = np.repeat(np.array([range(n)]).T, axis=1, repeats=2)
    P, inds, d = picking_algorithm(X, m)
    assert (P == X[inds, :]).all
    assert (d == dist.cdist(P, P)).all()
    assert np.amin(d + np.identity(m) * 10) > 2
