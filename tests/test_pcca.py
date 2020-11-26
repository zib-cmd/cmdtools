from cmdtools import utils
from cmdtools.analysis import pcca
from cmdtools.analysis import schur
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


def test_scipyschur(n=10, m=3):
    T = utils.randompropagator(n)
    massmatrix = np.diag(np.ones(n))

    solver = pcca.ScipySchur()
    X1 = solver.solve(T, m)
    X2 = solver.solve(T, m, massmatrix)

    # check if X1 and X2 span the same space
    tol = 1e-12
    assert np.linalg.matrix_rank(np.hstack([X1, X2]), tol) == m


# test whether krylovschur is doing the same as scipyschur
def test_krylovschur(n=10, m=5, N=100):
    if schur.HAS_SLEPC:
        for i in range(N):
            A = utils.randompropagator(n, reversible=False)
            try:
                S = schur.scipyschur(A, m, onseperation="error")
            except RuntimeError:
                continue
            K = schur.krylovschur(A, m, onseperation="continue")
            R = np.linalg.matrix_rank(np.concatenate([S, K], axis=1), tol=1e-12)
            assert R == m


N_BENCHMARK = 1000
M_BENCHMARK = 10


def test_bench_scipyschur(benchmark, n=N_BENCHMARK, m=M_BENCHMARK):
    T = utils.randompropagator(n)
    solver = pcca.ScipySchur()
    benchmark(solver.solve, T, m)


def test_bench_scipyqz(benchmark, n=N_BENCHMARK, m=M_BENCHMARK):
    T = utils.randompropagator(n)
    massmatrix = np.diag(np.ones(n))
    solver = pcca.ScipySchur()
    benchmark(solver.solve, T, m, massmatrix)


def test_bench_krylovschur(benchmark, n=N_BENCHMARK, m=M_BENCHMARK):
    if not pcca.HAS_SLEPC:
        return
    T = utils.randompropagator(n)
    solver = pcca.KrylovSchur(onseperation="continue")
    benchmark(solver.solve, T, m)


def test_bench_krylovschursparse_dense(benchmark, n=N_BENCHMARK, m=M_BENCHMARK):
    if not pcca.HAS_SLEPC:
        return
    from scipy import sparse
    T = utils.randompropagator(n)
    T = sparse.csr_matrix(T)
    solver = pcca.KrylovSchur(onseperation="continue")
    benchmark(solver.solve, T, m)


def test_bench_krylovschursparse_sparse(benchmark, n=N_BENCHMARK, m=M_BENCHMARK, p=0.01):
    if not pcca.HAS_SLEPC:
        return
    from scipy import sparse
    T = sparse.random(n, n, p)
    T = np.array(np.identity(n) + T)
    T = utils.rowstochastic(T)
    T = sparse.csr_matrix(T)
    solver = pcca.KrylovSchur(onseperation="continue")
    benchmark(solver.solve, T, m)
