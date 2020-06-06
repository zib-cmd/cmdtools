import numpy as np
from scipy.linalg import schur, ordqz
from .optimization import inner_simplex_algorithm, optimize
import warnings

# TODO: find a better solution to this
try:
    import slepc  # noqa: F401
    USE_SLEPC = True
except ImportError:
    USE_SLEPC = False


def pcca(T, n, pi=None):
    X = schurvects(T, n, pi)
    A = inner_simplex_algorithm(X)
    if n > 2:
        A = optimize(X, A)
    chi = X.dot(A)
    return chi


def schurvects(T, n, pi=None):
    """
    compute the leading n schurvectors X
    wrt. the scalar product induced by pi
    (i.e. X^T D X = I with D=diag(pi))
    """
    if pi is None:
        dim = np.size(T, 1)
        pi = np.full(dim, 1/dim)

    # Solve the transformed problem
    # Xb = schur(Pb)
    # Pb = D^1/2 P D^-1/2, X = D^-1/2 Xb
    # Pb Xb = Xb S => PX = X S
    # Xb^T Xb = I  => X^T D X = I

    D = np.diag(np.sqrt(pi))
    Di = np.diag(np.sqrt(1/pi))
    Pb = D.dot(T).dot(Di)

    if USE_SLEPC:
        Xb = krylovschur(Pb, n)
    else:
        Xb = scipyschur(Pb, n)

    X = Di.dot(Xb)
    X[:, 0] = X[:, 0] * np.sign(X[0, 0]) # fix sign of first column
    assertstructure(X, pi)
    return X


def assertstructure(X, pi):
    I = np.identity(np.size(X, 1))
    D = np.diag(pi)
    XTDX = X.T.dot(D).dot(X)
    assert np.all(np.isclose(X[:, 0], 1))
    assert np.all(np.isclose(XTDX - I, 0))


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