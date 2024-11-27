from cmdtools import utils
from cmdtools.analysis import pcca
from cmdtools.analysis import schur
import numpy as np
import pytest


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
    with pytest.warns(RuntimeWarning):
        m = pcca.pcca(T, 3)
    m = utils.order_membership(m)
    expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]
    assert np.isclose(m, expected).all()

def test_predefined_eigenvectors():
    X = np.random.rand(10,3)
    X[:,0] = 1
    pcca.PCCA(eigenvectors=X, n=2)
