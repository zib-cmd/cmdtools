from cmdtools import utils
from cmdtools.analysis import pcca
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
    T = utils.randompropagator(n, reversible=False)
    pcca.schurvects(T, m)


def test_schurvects_generalized(n=10, m=3):
    T = utils.randompropagator(n)
    X1 = pcca.schurvects(T, m)
    X2 = pcca.scipyschur(T, m, massmatrix=np.diag(np.ones(n)))

    # check if X1 and X2 span the same space
    assert np.linalg.matrix_rank(np.hstack([X1, X2])) == m


# test whether krylovschur is doing the same as scipyschur
def test_krylovschur(n=30, m=5, N=100):
    if pcca.USE_SLEPC:
        for i in range(N):
            A = utils.randompropagator(n, reversible=False)
            try:
                S = pcca.scipyschur(A, m, onseperation="error")
            except RuntimeError:
                continue
            K = pcca.krylovschur(A, m)
            R = np.linalg.matrix_rank(np.concatenate([S, K], axis=1), tol=1e-6)
            assert R == m
