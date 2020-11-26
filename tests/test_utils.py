import cmdtools.utils as utils
import numpy as np

def test_euclidean_torus():
    periods = [1, np.inf]
    x = np.array([-.5, 0])
    y = np.array([.5, 3])
    metric = utils.euclidean_torus(periods)
    assert metric(x,y) == 3