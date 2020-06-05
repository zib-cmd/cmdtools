import numpy as np
from scipy.linalg import schur, ordqz
from .optimization import inner_simplex_algorithm, optimize

# TODO: find a better solution to this
try:
    import slepc  # noqa: F401
    USE_SLEPC = True
except ImportError:
    USE_SLEPC = False


def pcca(T, n):
    X = schurvects(T, n)
    A = inner_simplex_algorithm(X)
    if n > 2:
        A = optimize(X, A)
    chi = X.dot(A)
    return chi


def schurvects(T, n):
    if USE_SLEPC:
        return krylovschur(T, n)
    else:
        return scipyschur(T, n)


def scipyschur(T, n, massmatrix=None):
    e = np.sort(np.linalg.eigvals(T))

    v_in  = np.real(e[-n])
    v_out = np.real(e[-(n + 1)])

    # do not seperate conjugate eigenvalues
    assert not np.isclose(v_in, v_out), \
        "Cannot seperate conjugate eigenvalues, choose another n"

    # determine the eigenvalue gap
    cutoff = (v_in + v_out) / 2

    # schur decomposition
    if massmatrix is None:
        _, X, _ = schur(T, sort=lambda x: np.real(x) > cutoff)
    else:
        _, _, _, _, _, X = \
            ordqz(T, massmatrix, sort=lambda a, b: np.real(a / b) > cutoff)

    return X[:, 0:n]  # use only first n vectors


def krylovschur(A, n):
    from petsc4py import PETSc
    from slepc4py import SLEPc
    M = PETSc.Mat().create()
    M.createDense(list(np.shape(A)), array=A)
    E = SLEPc.EPS().create()
    E.setOperators(M)
    E.setDimensions(nev=n)
    E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    E.solve()
    X = np.column_stack([x.array for x in E.getInvariantSubspace()])
    return X[:, :n]
