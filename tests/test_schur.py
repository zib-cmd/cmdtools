from cmdtools import utils, schur, sqra, diffusion
import numpy as np
from scipy.linalg import expm
import pytest

N_BENCHMARK = 2000
M_BENCHMARK = 5

u = np.random.rand(N_BENCHMARK)
Q_sparse = sqra.SQRA(u).Q
Q_sparse = diffusion.TripleWell(nx=50, ny=20).Q
Q_dense  = Q_sparse.toarray()
T_dense  = expm(Q_sparse)


def test_scipyschur(n=10, m=3):
    T = utils.randompropagator(n)
    massmatrix = np.diag(np.ones(n))

    solver = schur.ScipySchur()
    X1 = solver.solve(T, m)
    X2 = solver.solve(T, m, massmatrix)

    # check if X1 and X2 span the same space
    tol = 1e-12
    assert np.linalg.matrix_rank(np.hstack([X1, X2]), tol) == m


@pytest.mark.benchmark(group="schur")
def test_bench_scipyschur(benchmark, n=N_BENCHMARK, m=M_BENCHMARK):
    solver = schur.ScipySchur()
    benchmark(solver.solve, Q_dense, m)


@pytest.mark.benchmark(group="schur")
def test_bench_scipyqz(benchmark):
    n = np.size(Q_sparse, axis=1)
    massmatrix = np.diag(np.ones(n))
    solver = schur.ScipySchur()
    benchmark(solver.solve, Q_dense, M_BENCHMARK, massmatrix)


if schur.HAS_SLEPC:
    # test whether krylovschur is doing the same as scipyschur
    def test_krylovschur(n=10, m=5, N=100):
        for i in range(N):
            A = utils.randompropagator(n, reversible=False)
            try:
                S = schur.scipyschur(A, m, onseperation="error")
            except RuntimeError:
                continue
            K = schur.krylovschur(A, m, onseperation="continue")
            R = np.linalg.matrix_rank(np.concatenate([S, K], axis=1), tol=1e-12)
            assert R == m

    @pytest.mark.benchmark(group="schur")
    def test_bench_krylovschur_dense(benchmark):
        solver = schur.KrylovSchur(onseperation="continue")
        benchmark(solver.solve, Q_dense, M_BENCHMARK)

    @pytest.mark.benchmark(group="schur")
    def test_bench_krylovschur_sparse(benchmark):
        solver = schur.KrylovSchur(onseperation="continue")
        benchmark(solver.solve, Q_sparse, M_BENCHMARK)
