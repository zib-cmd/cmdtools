import numpy as np
from scipy.optimize import fmin


def inner_simplex_algorithm(X):

    # the ISA algorithm assumes the first column to be the constant eigenvector
    # which we construct by projection
    proj = np.dot(np.ones(np.shape(X)[0]), X)
    assert not(np.isclose(proj[0], 0))  # keep subspace of first column
    X[:, 0] = np.dot(X, proj)

    assert all(np.isclose(X[0, 0], X[:, 0]))
    X[:, 0] = 1
    i = indexsearch(X)
    return np.linalg.inv(X[i, :])


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


def optimize(X, A, maxiter=1000):
    if np.size(X, axis=1) > 1:
        x = A[1:, 1:]
        x = fmin(objective, x0=x, args=(X, A), maxiter=maxiter)
        n = np.size(A, axis=1) - 1
        A[1:, 1:] = x.reshape(n, n)
        fillA(A, X)
    return A


def objective(alpha, X, A):
    n = np.size(X, axis=1) - 1
    A[1:, 1:] = alpha.reshape(n, n)
    fillA(A, X)
    return -np.trace(np.diag(1 / A[0, :]).dot(A.T).dot(A))


def fillA(A, X):
    """ Converts the given matrix into a feasible transformation matrix. """
    A[1:, 0] = -np.sum(A[1:, 1:], axis=1)  # row-sum condition
    A[0, :]  = -np.min(X[:, 1:].dot(A[1:, :]), axis=0)  # maximality condition
    A /= np.sum(A[0, :])  # rescale to feasible set
