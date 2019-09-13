import pcca
import utils
import numpy as np


def test_random():
    T = utils.randompropagator(10)
    m = pcca.pcca(T, 3)
    assert utils.isstochastic(m)


def test_example():
    T = utils.example_metastab4()
    m = pcca.pcca(T, 2)
    expected = [[1, 0],
                [.5, .5],
                [0, 1],
                [0, 1]]
    assert np.isclose(utils.order_membership(m), expected).all()


def test_schurvects():
    T = utils.randompropagator(10)
    X = pcca.schurvects(T, 4)
    # first eigenvector is 1
    assert (X[:, 0] == 1).all()
