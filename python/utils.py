import numpy as np


def randompropagator(n, reversible=True):
    T = np.random.rand(n, n)
    if reversible:
        T = (T + T.T)
    T /= T.sum(axis=1)[:, None]  # rownormalize
    return T


def example_metastab4():
    T = [[.9, .1, 0, 0],
         [.5, 0, .25, .25],
         [0, .1, 0, .9],
         [0, .1, .9, 0]]
    return T


def isstochastic(P):
    return np.isclose(P.sum(axis=1), 1).all()


def order_membership(m):
    """ order the membership matrix, `first comes first` """
    return m[:, np.argsort(np.argmax(m, axis=0))]
