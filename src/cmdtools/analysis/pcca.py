import numpy as np
from .optimization import Optimizer
from ..utils import get_pi
from .schur import ScipySchur, Eigenvectors
import warnings


class PCCA:
    def __init__(self, T=None, n=None, pi="uniform", massmatrix=None,
                 eigensolver=ScipySchur(), optimizer=Optimizer(), eigenvectors=None):
        if eigenvectors is not None:  # hacky way to insert precomputed eigenvectors
            pi = get_pi(eigenvectors, pi)
            eigensolver = Eigenvectors(eigenvectors)

        self.T = T
        self.n = n
        self.pi = get_pi(T, pi)
        self.massmatrix = massmatrix
        self.eigensolver = eigensolver
        self.optimizer = optimizer
        if T is not None or eigenvectors is not None:
            self.solve()

    def solve(self):
        T, n, pi, massmatrix, eigensolver, optimizer = self.T, self.n, \
            self.pi, self.massmatrix, self.eigensolver, self.optimizer

        X = eigensolver.solve(T, n, massmatrix)
        X = gramschmidt(X, pi)
        A = optimizer.solve(X, pi)
        chi = np.dot(X, A)

        self.chi, self.X, self.A = chi, X, A


def pcca(T, n, **kwargs):
    return PCCA(T, n, **kwargs).chi


def gramschmidt(X, pi):
    """Gram Schmidt orthogonalization wrt. scalar product induced by pi"""
    X = np.copy(X)
    if np.isclose(np.dot(X[:, 0], np.full(np.size(X, 0), 1)), 0):
        raise RuntimeError("First column is orthogonal to 1-Vector, \
                            try swapping the columns")
    for i in range(np.size(X, 1)):
        if i == 0:
            X[:, 0] = np.sqrt(1 / sum(pi))
        else:
            X[:, i] -= X[:, :i] @ (X[:, i]*pi @ X[:, :i])
            X[:, i] /= np.sqrt((X[:, i]*pi @ X[:, i]))
    return X
