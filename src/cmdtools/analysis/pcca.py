import numpy as np
from .optimization import Optimizer
from ..utils import get_pi
from .schur import ScipySchur
import warnings


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
