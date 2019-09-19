import numpy as np
from scipy.linalg import schur, ordqz
from optimization import inner_simplex_algorithm, optimize


def pcca(T, n):
    X = schurvects(T, n)
    A = inner_simplex_algorithm(X)
    if n > 2:
        A = optimize(X, A)
    chi = X.dot(A)
    return chi


def schurvects(T, n, massmatrix=None):
    e = np.sort(np.linalg.eigvals(T))

    # do not seperate conjugate eigenvalues
    assert not np.isclose(np.real(e[-n]), np.real(e[-(n+1)])), \
        "Cannot seperate conjugate eigenvalues, choose another n"

    # determine the eigenvalue gap
    cutoff = (e[-n] + e[-(n + 1)]) / 2

    # schur decomposition
    if massmatrix is None:
        _, X, _ = schur(T, sort=lambda x: x > cutoff)
    else:
        _, _, _, _, _, X = \
            ordqz(T, massmatrix, sort=lambda a,b: a/b > cutoff)

    X = X[:, 0:n]  # use only first n vectors

    # move constant vector to the front, make it 1
    X /= np.linalg.norm(X, axis=0)
    i = np.argmax(np.abs(np.sum(X, axis=0)))
    X[:, i] = X[:, 0]
    X[:, 0] = 1

    return X
