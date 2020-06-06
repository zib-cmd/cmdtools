import numpy as np
from scipy.optimize import fmin

""" References:

For the general optimization:
Deuflhard, P. and Weber, M., 2005. Robust Perron cluster analysis in conformation dynamics. https://doi.org/10.1016/j.laa.2004.10.026
For the objective function:
Röblitz, S. and Weber, M., 2013. Fuzzy spectral clustering by PCCA+: https://doi.org/10.1007/s11634-013-0134-6
"""


def inner_simplex_algorithm(X):
    """
    Return the transformation A mapping those rows of X
    which span the largest simplex onto the unit simplex.
    """
    ind = indexsearch(X)
    return np.linalg.inv(X[ind, :])


def indexsearch(X):
    """ Return the indices to the rows spanning the largest simplex """

    k = np.size(X, axis=1)
    X = X.copy()

    ind = np.zeros(k, dtype=int)
    for j in range(0, k):
        # find the largest row
        rownorm = np.linalg.norm(X, axis=1)
        ind[j] = np.argmax(rownorm)

        if j == 0:
            # translate to origin
            X -= X[ind[j], :]
        else:
            # remove subspace of this row
            X /= rownorm[ind[j]]
            v  = X[ind[j], :]
            X -= np.outer(X.dot(v), v)

    return ind


def optimize(X, A, pi, maxiter=1000):
    """
    optimization of A
    - the feasiblization routine fillA() requires
    the first column of X to be the constant one-vector
    - the optimzation criterion expects X^T D X = I
    (where D is the stationary diagonal matrix)
    """
    assertstructure(X, pi)
    x = A[1:, 1:]
    x = fmin(objective, x0=x, args=(X, A), maxiter=maxiter)
    n = np.size(A, axis=1) - 1
    A[1:, 1:] = x.reshape(n, n)
    fillA(A, X)
    return A


def assertstructure(X, pi):
    I = np.identity(np.size(X, 1))
    D = np.diag(pi)
    XTDX = X.T.dot(D).dot(X)
    assert np.all(np.isclose(X[:, 0], 1))
    assert np.all(np.isclose(XTDX - I, 0))


def objective(alpha, X, A):
    """ Equation (16) from Röblitz, Weber (2013) """
    n = np.size(X, axis=1) - 1
    A[1:, 1:] = alpha.reshape(n, n)
    fillA(A, X)
    return -np.trace(np.diag(1 / A[0, :]).dot(A.T).dot(A))


def fillA(A, X):
    """
    Converts the given matrix into a feasible transformation matrix.
    Algorithm 3.10 from Weber (2006)
    """
    A[1:, 0] = -np.sum(A[1:, 1:], axis=1)  # row-sum condition
    A[0, :]  = -np.min(X[:, 1:].dot(A[1:, :]), axis=0)  # maximality condition
    A /= np.sum(A[0, :])  # rescale to feasible set
