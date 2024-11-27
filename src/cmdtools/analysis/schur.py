import numpy as np
import warnings
from scipy.linalg import schur, ordqz
from scipy import sparse
from ..utils import is_generator, is_rowstochastic

# TODO: find a better solution to this
try:
    import slepc4py  # noqa: F401
    HAS_SLEPC = True
except ImportError:
    HAS_SLEPC = False

DEFAULT_WHICH = "auto"
DEFAULT_ONSEPERATION = "warn"


def parse_which(A, which):
    if which != "auto":
        return which
    if is_rowstochastic(A):
        return "LM"
    elif is_generator(A):
        return "LR"
    else:
        raise(ValueError("Given matrix is neither P nor Q matrix"))

class Eigenvectors:
    """ A class used to precomputed Eigenvectors. """
    def __init__(self, X):
        self.X = X

    def solve(self, A, n, massmatrix=None):
        assert massmatrix is None
        return self.X[:, 0:n]


class KrylovSchur:
    def __init__(self, onseperation="continue", which=DEFAULT_WHICH, maxiter=1000, tolerance=1e-6):
        self.onseperation = onseperation
        self.which = which
        self.maxiter = maxiter
        self.tolerance = tolerance

    def solve(self, A, n, massmatrix=None):
        return krylovschur(A, n, massmatrix, self.onseperation, self.which, self.tolerance, self.maxiter)


class ScipySchur:
    def __init__(self, onseperation=DEFAULT_ONSEPERATION, which=DEFAULT_WHICH):
        self.onseperation = onseperation
        self.which = which

    def solve(self, A, n, massmatrix=None):
        return scipyschur(A, n, massmatrix, self.onseperation, self.which)


def scipyschur(A, n, massmatrix=None, onseperation=DEFAULT_ONSEPERATION, which=DEFAULT_WHICH):
    which = parse_which(A, which)
    if which == "LM":
        def sortfun(x):
            return np.abs(x)
    elif which == "LR":
        def sortfun(x):
            return np.real(x)
    else:
        raise NotImplementedError("the choice of `which` is not supported")

    if sparse.issparse(A):
        A = A.toarray()

    e = np.sort(sortfun(np.linalg.eigvals(A)))
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
            return scipyschur(A, n+1, massmatrix, "error")
        else:
            raise RuntimeError(msg)

    # determine the eigenvalue gap
    cutoff = (v_in + v_out) / 2

    # schur decomposition
    if massmatrix is None:
        # NOTE: assuming that the sort callable is passed complex numbers as (a,b) is undocumented behaviour
        _, X, _ = schur(A, sort=lambda a, b: sortfun(complex(a, b)) > cutoff)
    else:
        _, _, _, _, _, X = \
            ordqz(A, massmatrix, sort=lambda a, b: sortfun(a/b) > cutoff)

    return X[:, 0:n]  # use only first n vectors


def krylovschur(A, n, massmatrix=None, onseperation=DEFAULT_ONSEPERATION, which=DEFAULT_WHICH, tolerance=1e-6, maxiter=100):
    which = parse_which(A, which)

    if massmatrix is not None:
        raise NotImplementedError
    if onseperation != "continue":
        raise NotImplementedError

    from slepc4py import SLEPc
    M = petsc_matrix(A)
    E = SLEPc.EPS().create()
    E.setOperators(M)
    E.setDimensions(nev=n)
    E.setConvergenceTest(E.Conv.ABS)
    E.setTolerances(tolerance, maxiter)
    if which == "LR":
        E.setWhichEigenpairs(E.Which.LARGEST_REAL)
    elif which == "LM":
        E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE)
    else:
        raise NotImplementedError("the choice of `which` is not supported")
    E.solve()
    if E.getConverged() < n:
        raise RuntimeError("Schur decomposition did not converge, consider more 'maxiter' or a higher 'tolerance'")
    X = np.column_stack([x.array for x in E.getInvariantSubspace()])
    return X[:, :n]


def petsc_matrix(A):

    from petsc4py import PETSc

    M = PETSc.Mat()
    if sparse.issparse(A):
        A = sparse.csr_matrix(A)
        nrows = np.size(A, 0)
        M.createAIJWithArrays(nrows, (A.indptr, A.indices, A.data))
    else:
        M.createDense(list(np.shape(A)), array=A)
    return M
