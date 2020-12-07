import cmdtools.utils as utils
import numpy as np


def test_euclidean_torus():
    periods = [1, np.inf]
    x = np.array([-.5, 0])
    y = np.array([.5, 3])
    metric = utils.euclidean_torus(periods)
    assert metric(x, y) == 3


def test_torus_minus():
    z = np.array([0])  # zero array to cast everyhing to arrays
    for s in [0, 4]:
        assert np.isclose(utils.torus_minus(s, 0+z, 1+z), 0)
        assert np.isclose(utils.torus_minus(.1+s, .4+z, 1+z), -0.3)
        assert np.isclose(utils.torus_minus(.4+s, .1+z, 1+z), 0.3)
        assert np.isclose(utils.torus_minus(.1+s, .9+z, 1+z), 0.2)
        assert np.isclose(utils.torus_minus(.9+s, .1+z, 1+z), -0.2)
