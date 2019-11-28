import numpy as np
from numpy.random import rand
from cmdtools import utils
from cmdtools.estimation import galerkin


def test_propagator():
    x = rand(10, 3)
    centers = rand(4, 3)
    traj = galerkin.Trajectory(timeseries=x, centers=centers)
    P = traj.propagator
    assert utils.is_rowstochastic(P)

def test_propagator_identity():
    x = np.ones([10,3])
    centers = rand(4,3)
    traj = galerkin.Trajectory(timeseries=x, centers=centers)
    P = traj.propagator
    assert np.isclose(P, np.identity(4)).all()
