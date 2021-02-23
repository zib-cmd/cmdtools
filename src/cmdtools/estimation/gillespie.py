import numpy as np
from . import voronoi


def gillespie(Q, x0, n_iter):
    x = x0
    xs = [x]
    ts = [0]
    for i in range(n_iter):
        rate = np.sum(Q[x,:]) - Q[x,x]
        tau = np.random.exponential(1/rate)
        q = Q[x,:] / rate
        q[x] = 0
        x = np.random.choice(range(len(q)), p=q)

        ts.append(ts[-1]+tau)
        xs.append(x)

    return np.array(xs), np.array(ts)


def stroboscopic_inds(ts):
    """ given a sorted array of times, return the indices to the last times before each unit time step """
    return(np.searchsorted(ts, np.arange(np.max(ts)+1),side="right")-1)


def propagator(xs, ts, nstates):
    xs_strob = xs[stroboscopic_inds(ts)]
    return voronoi.propagator(xs_strob, nstates, 1)
