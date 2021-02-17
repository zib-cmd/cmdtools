import numpy as np
import scipy.sparse as sp


def sqra(u, A, beta, phi):
    """ Square-root approximation of the generator
    (of the Overdamped Langevin model)

    u: vector of pointwise evaluation of the potential
    A: adjacency matrix of the discretization
    beta: inverse temperature
    phi: the flux constant, determined by the temperature and the discr.
    """

    pi  = np.sqrt(np.exp(- beta * u))  # Boltzmann distribution
    pi /= np.sum(pi)

    D  = sp.diags(pi)
    D1 = sp.diags(1 / pi)
    Q  = phi * D1 @ A @ D
    Q  = Q - sp.diags(np.array(Q.sum(axis=1)).flatten())
    if sp.issparse(Q):
        Q = Q.tocsc()
    return Q


def adjacency_nd(dims, torus=False):
    nd = len(dims)
    N = np.prod(dims)

    neighbours = np.vstack([np.diag(np.ones(nd, dtype=int)), -np.diag(np.ones(nd, dtype=int))])
    singletondim = np.array(dims) == 1
    neighbours[:, singletondim] = 0

    row = np.zeros(N*2*nd, dtype=int)
    col = np.zeros(N*2*nd, dtype=int)
    data = np.ones(N*2*nd, dtype=bool)
    cut = np.zeros(N*2*nd, dtype=bool)

    for i in range(N):
        multi = np.unravel_index(i, dims)  # find multiindex of current cell
        mn = multi + neighbours  # add neighbour offset
        if not torus:
            cut[i*2*nd:(i+1)*2*nd] = np.sum((mn < 0) + (mn >= dims), axis=1)  # check if boundary is hit
        mn = np.mod(multi + neighbours, dims)  # glue together boundary
        neighbour_flat = np.ravel_multi_index(mn.T, dims)  # back to flat inds
        # print(neighbour_flat)
        row[i*2*nd:(i+1)*2*nd] = i
        col[i*2*nd:(i+1)*2*nd] = neighbour_flat

    if not torus:  # cut out the points which were glued at boundaries
        sel = ~cut
        data = data[sel]
        row = row[sel]
        col = col[sel]

    A = sp.coo_matrix((data, (row, col)))
    A.setdiag(0)
    return A.tocsr()


class SQRA:
    """ approximate the generator/rate-matrix Q of the Overdamped-Langevin dynamics
    Args:
        u (ndarray): The potential function evaluated at the grid points.
        beta (float): The inverse temperatur of the system (scales the rates nonlinear).
        phi (float): Linear scaling factor of the rates (depending on the grid volume).
        A (matrix, optional): Adjacency matrix of the grid. If left empty, automatically compute it based on the shape of `u`.
        torus(list of bool): Whether to glue the corresponding dimensions together at their resp. boundaries. Only used in the automatic generation of `A`.

    Attributes:
        A: Adjacency matrix
        Q: The computed generator
    """

    def __init__(self, u, beta=1, phi=1, A=None, torus=False):
        self.u = u
        self.beta = beta
        self.phi = phi

        self.dims = np.shape(u)

        self.A = adjacency_nd(self.dims, torus) if A is None else A
        self.Q = self.sqra()
        self.N = self.Q.shape[0]

    def sqra(self):
        return sqra(self.u.flatten(), self.A, self.beta, self.phi)


# conversion between coldness (beta) and epsilon in sde formulation
# necessary for the computation of phi
def beta_to_epsilon(b):
    return 2 / b
