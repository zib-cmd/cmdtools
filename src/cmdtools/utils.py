import numpy as np


def randompropagator(n, reversible=True):
    T = np.random.rand(n, n)
    if reversible:
        T = (T + T.T)
    return rowstochastic(T)


def rowstochastic(T):
    return T / T.sum(axis=1)[:, None]


def example_metastab4():
    """
    Return the transition matrix for a simple metastable model.
    The 4 correpsonding states are [A,T,B,B],
    where A and B are metastable and T is a transition region
    """
    T = [[.9, .1, 0, 0],     # metastable A
         [.5, 0, .25, .25],  # transition
         [0, .1, 0, .9],     # metastable B
         [0, .1, .9, 0]]     # metastable B
    return T


def is_rowstochastic(P):
    return np.isclose(P.sum(axis=1), 1).all() and \
        (P >= -1e-12).all()


def order_membership(m):
    """ order the membership matrix, `first comes first` """
    return m[:, np.argsort(np.argmax(m, axis=0))]
