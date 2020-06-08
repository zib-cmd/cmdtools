import numpy as np
from scipy.linalg import schur, ordqz
from .optimization import inner_simplex_algorithm, optimize
from ..utils import get_pi
import warnings

# TODO: find a better solution to this
try:
    import slepc  # noqa: F401
    USE_SLEPC = True
except ImportError:
    USE_SLEPC = False


def pcca(T, n, pi="uniform"):
    pi = get_pi(T, pi)
    X = schurvects(T, n, pi)
    A = inner_simplex_algorithm(X)
    if n > 2:
        A = optimize(X, A, pi)
    chi = X.dot(A)
    return chi


def schurvects(T, n, pi):
    """
    compute the leading n schurvectors X
    wrt. the scalar product induced by pi
    (i.e. X^T D X = I with D=diag(pi))
    """

    if USE_SLEPC:
        X = krylovschur(T, n)
    else:
        X = scipyschur(T, n)

    X = gramschmidt(X, pi)
    return X


def gramschmidt(X, pi):
    """Gram Schmidt orthogonalization wrt. scalar product induced by pi"""
    X = np.copy(X)
    D = np.diag(pi)
    for i in range(np.size(X, 1)):
        if i == 0:
            if np.isclose(np.dot(X[:, 0], np.full(np.size(X, 0), 1)), 0):
                # this should not happen, if so we have to swap columns
                raise RuntimeError("First column is orthogonal to 1-Vector")
            X[:, 0] = 1
        else:
            for j in range(i):
                X[:, i] -= X[:, i].dot(D).dot(X[:, j]) * X[:, j]
            X[:, i] /= np.sqrt(X[:, i].dot(D).dot(X[:, i]))
    return X


def scipyschur(T, n, massmatrix=None, onseperation="warn"):
    e = np.sort(np.linalg.eigvals(T))

    v_in  = np.real(e[-n])
    v_out = np.real(e[-(n + 1)])

    # do not seperate conjugate eigenvalues
    if np.isclose(v_in, v_out):
        msg = "Invalid number of clusters (splitting conjugate eigenvalues, choose another n)"
        if onseperation == "warn":
            warnings.warn(msg, RuntimeWarning)
        elif onseperation == "continue":
            pass
        elif onseperation == "fix":
            return scipyschur(T, n+1, massmatrix, "error")
        else:
            raise RuntimeError(msg)

    # determine the eigenvalue gap
    cutoff = (v_in + v_out) / 2

    # schur decomposition
    if massmatrix is None:
        _, X, _ = schur(T, sort=lambda x: np.real(x) > cutoff)
    else:
        _, _, _, _, _, X = \
            ordqz(T, massmatrix, sort=lambda a, b: np.real(a / b) > cutoff)

    return X[:, 0:n]  # use only first n vectors


def krylovschur(A, n, massmatrix=None, onseperation="continue"):
    if massmatrix is not None:
        raise NotImplementedError
    if onseperation != "continue":
        raise NotImplementedError

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


def normalizeschur(X):
    # find the constant eigenvector corresponding to ev. 1,
    # move it to the front and set it to 1
    # as required by the optimization routine

    X /= np.linalg.norm(X, axis=0)
    i = np.argmax(np.abs(np.sum(X, axis=0)))
    X[:, i] = X[:, 0]
    X[:, 0] = 1  # TODO: check if this column is indeed constant

    return X


def normalizeschur2(X):
    n, m = np.shape(X)
    T = np.identity(m)
    T[:, 0] = np.dot(np.ones(n)/np.sqrt(n), X)
    if np.isclose(T[0, 0], 0):
        raise RuntimeError("X[:,1] must not be orthogonal to the one-vector")
    tX = np.dot(X, T)
    return tX
