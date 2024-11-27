import numpy as np


def randompropagator(n, reversible=True):
    T = np.random.rand(n, n)
    if reversible:
        T = (T + T.T)
    return rowstochastic(T)


def get_pi(T, pi="uniform"):
    if isinstance(pi, str) and pi == "uniform":
        dim = np.size(T, 0)
        pi = np.full(dim, 1./dim)
    elif isinstance(pi, str) and pi == "auto":
        raise NotImplementedError  # TODO Eigenvector to EV 1
    assert np.isclose(np.sum(pi), 1)
    return pi


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
         [0, .1, .5, .4],     # metastable B
         [0, .1, .4, .5]]     # metastable B
    return np.array(T)


def is_rowstochastic(P):
    return np.isclose(P.sum(axis=1), 1).all() and \
        (P >= -1e-12).all()


def is_generator(Q):
    """ check necessary conditions for Q being a generator (not sufficient) """
    return np.allclose(Q.sum(axis=1), 0)


def order_membership(m):
    """ order the membership matrix, `first comes first` """
    return m[:, np.argsort(np.argmax(m, axis=0))]


def euclidean_torus(periods):
    """ euclidean metric on a d-dimensional torus with given periods.
    using np.inf for a specific period corresponds to the usual euclidean distance in that dimension """
    def metric(x,y):
        d = x-y
        d = np.minimum(d % periods, -d % periods)
        return np.sqrt(np.sum(d**2))
    return metric


def torus_minus(x, y, periods):
    """ compute the shortest vector from y to x on the topological torus """
    d = (x-y)
    fin = periods < np.inf  # only modulo the glued dimensions
    d[fin] = d[fin] % periods[fin]  # distance on "the front"
    d = d - (d > periods / 2) * periods  # distance on "the back"
    return d
