"""
sparse implementation of the Galerkin discretization to the AJC as in the preprint.

also provides finite hitting time computations via the space-time committor approach.
"""


import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import spsolve
from copy import deepcopy
from scipy.optimize import minimize


class AJC():
    """ sparse implementation of the Galerkin discretization to the AJC as in the preprint
    uses the """

    def __init__(self, Q, dt):
        self.nx = Q[0].shape[0]
        self.nt = len(dt)
        self.nxt = self.nx * self.nt
        self.Q = Q
        self.dt = np.array(dt)
        self.compute()

    def compute(self):
        self.qt, self.qi = self.qtilde(self.Q)
        self.k, self.S = self.jumpkernel(self.qt, self.qi, self.dt)

    @staticmethod
    def jumpkernel(qt, qi, dt):
        """ compute the galerkin discretization of the jump kernel eq[50] """

        nt, nx = np.shape(qi)
        s = np.exp(- dt[:, None] * qi)  # (nt, nx)

        S = np.zeros((nt, nt, nx))  # temporal jump chain
        J = np.empty((nt, nt), dtype=object)

        for k in range(nt):

            prod = np.vstack((np.ones(nx), np.cumprod(s[k+1:-1,:], axis=0)))
            if dt[k] > 0:
                factor = (1 - s[k,:]) / (dt[k] * qi[k])
            else:  # limit for dt -> 0
                factor = np.ones(nx)

            S[k, k+1:, :] = factor * (1-s[k+1:,:]) * prod
            S[k, k, :]    = 1 - factor

        for k in range(nt):
            for l in range(nt):
                J[k, l] = sp.diags(S[k, l, :]).dot(qt[l])  # is  csr_scale_rows faster?
        return J, S

    @staticmethod
    def qtilde(Qs):
        """ given an array of rate matrices compute the jump chain matrix
        qt[t,i,j] = q^tilde_ij(t) : the row normalized jump matrix [eq. 14]
        qi[t,i] = q_i(t): the outflow rate [eq. 6]
        """

        qt = [sp.dok_matrix(Q) for Q in Qs]
        nt = len(qt)
        nx = qt[0].shape[0]
        qi = np.zeros((nt, nx))
        for k in range(nt):
            qt[k][range(nx), range(nx)] = 0
            qi[k, :] = qt[k].sum(axis=1).A[:, 0]
            qt[k] = sp.diags(1/qi[k, :]).dot(qt[k])
            z = np.where(qi[k, :] == 0)           # special case q_i = 0 => q_ij = kron_ij
            qt[k][z[0], :] = 0
            qt[k][z[0], z[0]] = 1

        return qt, qi

    @property
    def km(self):
        if not hasattr(self, "_km"):
            self._km = flatten_spacetime(self.k)  # TODO: find a good way to deal with differing representations
        return self._km

    # TODO: compute this without falling back to the kernel matrix
    def jump(self, p):
        return np.reshape(p.flatten() @ self.km, p.shape)

    def geiger(self, p, n=100):
        g = np.copy(p)
        pp = p
        for i in range(n):
            pp = self.jump(pp)
            g = g + pp
        return g

    def holding_probs(self):
        S = (1 - np.cumsum(self.S, axis=1))
        S = np.moveaxis(np.triu(np.moveaxis(S, 2, 0)), 0, 2)  # set entries below (from-to-)diagonal to 0
        return S

    def synchronize(self, p):
        return np.einsum('sti, is -> it', self.holding_probs(), p)

    # SOLVERS FOR THE LINEAR SYSTEM
# class AugmentedSolver:
#     """ methods for solving the augmented linear system
#     Ax = b where A is upper triangular block
#     where K[i,j] containing the blocks and b[i] the corresponding RHS """

#     def __init__(self, K, holdingprobs):
#         self.K = K
#         self.holdingprobs = holdingprobs
#         self.nt = np.size(K, axis=0)
#         self.nx = np.size(K[0,0], axis=0)

    @classmethod
    def backwardsolve(self, K, b):
        nt = np.size(K, axis=0)
        b = b.copy()
        q = np.zeros(np.shape(b))

        for s in range(nt)[::-1]:
            for t in np.arange(s, nt)[::-1]:
                if s < t:
                    b[s] -= K[s, t] @ q[t]
                elif s == t:
                    q[s] = spsolve(K[s, s], b[s])
        return q

    def koopman_system_one(self, i):
        """ solve the koopman system for a single target state """
        nx, nt = self.nx, self.nt

        K = - deepcopy(self.k)
        b = np.zeros((nt, nx))
        S = self.holding_probs()[:, -1, :]

        for s in range(nt):
            K[s, s] += sp.identity(nx)
            b[s][i] = S[s, i]

        K[-1, -1] = sp.identity(nx)
        b[-1][:] = 0
        b[-1][i] = 1

        q = self.backwardsolve(K, b)
        return q

    def koopman_system_iterated(self):
        """ solve the koopman system for a all target states by solving individually for each """
        nx = self.nx
        K = np.zeros((nx, nx))
        for i in range(nx):
            K[:, i] = self.koopman_system_one(i)[0, :]
        return K

    def space_time_committor(self, g):
        r""" compute the space-time-committor
        .. math::
           q = /dagger J q + S g on /Omega
           q = g on /delta /Omega

        Args:
            g ( (nt) x (nx) array): boundary condition, set to np.nan in the interior of Omega
        """

        nx, nt = self.nx, self.nt

        # A = Id - K
        A = -deepcopy(self.k)
        for i in range(nt):
            A[i, i] += sp.identity(nx)

        b = np.zeros((nt, nx))
        S = self.holding_probs()

        bc_time = np.zeros(nx, int)
        bc_vals = np.zeros(nx)

        for t in reversed(range(nt)):
            bc_inds = np.isfinite(g[t, :])  # indices of currently active boundary cond.
            bc_time[bc_inds] = t           # update time of last  active boundary cond.
            bc_vals[bc_inds] = g[t, bc_inds]  # and the respective value

            b[t] = S[t, bc_time, range(nx)] * bc_vals  # contribution of staying into the next boundary

            for i in np.where(bc_inds):  # fix boundary values
                for tt in range(t, nt):
                    A[t, tt][i, :] = 0
                A[t, t][i, i] = 1
                b[t, i]      = g[t, i]

        q = self.backwardsolve(A, b)
        return q

    def koopman(self):
        return self.koopman_system()

    def koopman_system(self):
        return self.koopman_system_all()[0, :, :]

    def koopman_system_all(self):
        """ solve the koopman system for all target states together """
        nx, nt = self.nx, self.nt
        K = - deepcopy(self.k)
        b = np.zeros((nt, nx, nx))

        S = self.holding_probs()[:, -1, :]

        for s in range(nt):
            K[s, s] += sp.identity(nx)
            b[s] = np.diag(S[s, :])

        K[-1, -1] = sp.identity(nx, format="csr")

        q = self.backwardsolve(K, b)
        return q

    """ HITTING PROBABILITIES """

    def finite_time_hitting_prob(self, state):
        """ Compute the probability to hit a given state n_state over the
        time horizon of the jump chain from any space-time point by solving
        Kp = p in C, p=1 in C'
        where C' is the time-fibre of n_state and C the rest """
        nx, nt = self.nx, self.nt
        K = deepcopy(self.k)

        b = np.zeros((nt, nx))

        # Kp - p = 0 in C
        for s in range(nt):
            for t in np.arange(s, nt):
                K[s, t][state, :] = 0

            K[s, s] = K[s, s] - sp.identity(nx)
            K[s, s][state, state] = 1
            b[s, state] = 1

        q = self.backwardsolve(K, b)
        return q

    def finite_time_hitting_probs(self):
        """ finite_time_hitting_probs[i,j] is the probability to hit state j starting in i in the time window of the process """
        # TODO: check if this is indeed the ordering
        return np.vstack([self.finite_time_hitting_prob(i)[0, :] for i in range(self.nx)]).T

    def optimize(self, iters=100, penalty=0, x0=None, adaptive=False):
        """ gradient free optimization of the minimal finite time hitting prob """
        # TODO: assert identical sparsity structures
        j = self
        jp = deepcopy(self)
        if x0 is None:
            x0 = np.zeros_like(j.Q[0].data)

        def obj(x):
            self.lastx = x
            for t in range(j.nt):
                # jp.Q[t].data = np.maximum(j.Q[t].data + x, 0)  # TODO: ignore diagonal
                jp.Q[t].data = j.Q[t].data + x
            jp.compute()
            o = - min(jp.finite_time_hitting_probs())
            op = o + np.sum(np.abs(x)) ** 2 * penalty
            print(op)
            return op

            # q = sqrt (exp -bU / exp -bU)

        res = minimize(obj, x0=x0, method='nelder-mead', options={'maxiter': iters, 'adaptive': adaptive})
        obj(res.x)
        return res, jp


# TODO: this should also work for vectors
# we might want something to convert between stacked K[s,t][i,j], flattened [st, ij] and tensor [s,i,t,j]
def flatten_spacetime(tensor):
    ns, nt = tensor.shape
    for s in range(ns):
        for t in range(nt):
            if t == 0:
                row = tensor[s, t]
            else:
                row = sp.hstack((row, tensor[s, t]))
        if s == 0:
            M = row
        else:
            M = sp.vstack((M, row))
    return M
