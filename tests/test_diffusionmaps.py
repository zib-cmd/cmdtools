from cmdtools.estimation.diffusionmaps import DiffusionMaps, NNDistances
import numpy as np


def test_diffusionmaps():
    X = np.random.rand(100, 5)
    Y = np.random.rand(10, 5)
    dm1 = DiffusionMaps(X)
    dm2 = DiffusionMaps(X, distances=NNDistances(20))
    for dm in [dm1, dm2]:
        assert np.allclose(dm.oos_extension(X), dm.dms)
        dm.oos_extension(Y)
