import numpy as np
from scipy.linalg import schur
from optimization import inner_simplex_algorithm, optimize


def pcca(T, n):
    X = schurvects(T, n)
    A = inner_simplex_algorithm(X)
    if n > 2:
        A = optimize(X, A)
    chi = X.dot(A)
    return chi


def schurvects(T, n):
    # find prescribed eigenvalue "gap" for cutoff
    e = np.sort(np.linalg.eigvals(T))
    cutoff = (e[-n] + e[-(n + 1)]) / 2

    # schur decomposition
    _, X, nn = schur(T, sort=(lambda x: x > cutoff))
    assert nn == n, "cutting a schur block"  # dont cut schur blocks
    X = X[:, 0:n]  # use only first n vectors

    # move constant vector to the front, make it 1
    X /= np.linalg.norm(X, axis=0)
    i = np.argmax(np.abs(np.sum(X, axis=0)))
    X[:, i] = X[:, 0]
    X[:, 0] = 1

    return X
