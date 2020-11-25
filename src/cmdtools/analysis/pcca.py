import numpy as np
from scipy.linalg import schur, ordqz
from .optimization import Optimizer
from ..utils import get_pi
import warnings

# TODO: find a better solution to this
try:
    import slepc  # noqa: F401
    HAS_SLEPC = True
except ImportError:
    HAS_SLEPC = False

DEFAULT_WHICH = "LM"
DEFAULT_ONSEPERATION = "warn"

# Class interfaces
# These are mainly wrappers around the functions below


class KrylovSchur:
    def __init__(self, onseperation=DEFAULT_ONSEPERATION, which=DEFAULT_WHICH):
        self.onseperation = onseperation
        self.which = which

    def solve(self, A, n, massmatrix=None):
        return krylovschur(A, n, massmatrix, self.onseperation, self.which)


class ScipySchur:
    def __init__(self, onseperation=DEFAULT_ONSEPERATION, which=DEFAULT_WHICH):
        self.onseperation = onseperation
        self.which = which

    def solve(self, T, n, massmatrix=None):
        return scipyschur(T, n, massmatrix, self.onseperation, self.which)


class PCCA:
    def __init__(self, T=None, n=None, pi="uniform", massmatrix=None,
                 eigensolver=ScipySchur(), optimizer=Optimizer()):
        self.T = T
        self.n = n
        self.pi = get_pi(T, pi)
        self.massmatrix = massmatrix
        self.eigensolver = eigensolver
        self.optimizer = optimizer
        if T is not None:
            self.solve()

    def solve(self):
        T, n, pi, massmatrix, eigensolver, optimizer = self.T, self.n, \
            self.pi, self.massmatrix, self.eigensolver, self.optimizer

        chi, X, A, = \
            _pcca(T, n, pi, massmatrix, eigensolver, optimizer)

        self.chi, self.X, self.A = chi, X, A
        return chi


# Functions
# the logic of the above classes following a more functional style


# compatibility to old functioncalls / tests
def pcca(T, n, pi="uniform"):
    p = PCCA(T, n, pi)
    return p.solve()


# this will be the new pcca function, the old function is replaced by the Class
def _pcca(T, n, pi, massmatrix, eigensolver, optimizer):
    X = eigensolver.solve(T, n, massmatrix)
    X = gramschmidt(X, pi)
    A = optimizer.solve(X, pi)
    chi = np.dot(X, A)
    return chi, X, A


def gramschmidt(X, pi):
    """Gram Schmidt orthogonalization wrt. scalar product induced by pi"""
    X = np.copy(X)
    D = np.diag(pi)
    for i in range(np.size(X, 1)):
        if i == 0:
            if np.isclose(np.dot(X[:, 0], np.full(np.size(X, 0), 1)), 0):
                raise RuntimeError("First column is orthogonal to 1-Vector, \
                                    try swapping the columns")
            X[:, 0] = 1
        else:
            for j in range(i):
                X[:, i] -= X[:, i].dot(D).dot(X[:, j]) * X[:, j]
            X[:, i] /= np.sqrt(X[:, i].dot(D).dot(X[:, i]))
    return X


def scipyschur(T, n, massmatrix=None, onseperation="warn", which=DEFAULT_WHICH):

    if which == "LM":
        def sortfun(x):
            return np.abs(x)
    elif which == "LR":
        def sortfun(x):
            return np.real(x)
    else:
        raise NotImplementedError("the choice of `which` is not supported")

    e = np.sort(sortfun(np.linalg.eigvals(T)))
    v_in  = e[-n]
    v_out = e[-(n + 1)]

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
        # NOTE: assuming that the sort callable is passed complex numbers as (a,b) is undocumented behaviour
        _, X, _ = schur(T, sort=lambda a, b: sortfun(complex(a, b)) > cutoff)
    else:
        _, _, _, _, _, X = \
            ordqz(T, massmatrix, sort=lambda a, b: sortfun(a/b) > cutoff)

    return X[:, 0:n]  # use only first n vectors


def krylovschur(A, n, massmatrix=None, onseperation="continue", which=DEFAULT_WHICH):
    if massmatrix is not None:
        raise NotImplementedError
    if onseperation != "continue":
        raise NotImplementedError

    from slepc4py import SLEPc
    M = petsc_matrix(A)
    E = SLEPc.EPS().create()
    E.setOperators(M)
    E.setDimensions(nev=n)
    if which == "LR":
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    elif which == "LM":
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    else:
        raise NotImplementedError("the choice of `which` is not supported")
    E.solve()
    X = np.column_stack([x.array for x in E.getInvariantSubspace()])
    return X[:, :n]


def petsc_matrix(A):
    from scipy import sparse
    from petsc4py import PETSc

    M = PETSc.Mat()
    if sparse.issparse(A):
        A = sparse.csr_matrix(A)
        nrows = np.size(A, 0)
        M.createAIJWithArrays(nrows, (A.indptr, A.indices, A.data))
    else:
        M.createDense(list(np.shape(A)), array=A)
    return M
