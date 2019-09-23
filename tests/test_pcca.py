from pccap import utils
from pccap.analysis import pcca
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


def test_schurvects(n=10, m=4):
    T = utils.randompropagator(n)
    X = pcca.schurvects(T, m)
    # first eigenvector is 1
    assert (X[:, 0] == 1).all()


def test_schurvects_generalized(n=10, m=3):
    T = utils.randompropagator(n)
    X1 = pcca.schurvects(T, m)
    X2 = pcca.schurvects(T, m, massmatrix=np.diag(np.ones(n)))

    # check if X1 and X2 span the same space
    assert np.linalg.matrix_rank(np.hstack([X1, X2])) == m