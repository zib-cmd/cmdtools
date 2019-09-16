import pcca
import utils
import numpy as np


def test_random():
    T = utils.randompropagator(10)
    m = pcca.pcca(T, 3)
    assert utils.is_rowstochastic(m)


def test_example_n2():
    T = utils.example_metastab4()
    m = pcca.pcca(T, 2)
    m = utils.order_membership(m)
    expected = [[1, 0], [.5, .5], [0, 1], [0, 1]]
    assert np.isclose(m, expected).all()


def test_example_n3():
    T = utils.example_metastab4()
    m = pcca.pcca(T, 3)
    m = utils.order_membership(m)
    expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert np.isclose(m, expected).all()


def test_schurvects():
    T = utils.randompropagator(10)
    X = pcca.schurvects(T, 4)
    # first eigenvector is 1
    assert (X[:, 0] == 1).all()
