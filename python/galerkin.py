import utils
import numpy as np
from scipy.spatial import distance


def propagator(timeseries, centers, sigma):
    """ Given `timeseries` data, estimate the propagator matrix. 
    
    Uses the galerkin projection onto Gaussian ansatz functions
    with bandwidth `sigma` around the given `centers`. """
    m = get_membership(timeseries, centers, sigma)
    counts = m[0:-1, :].T @ m[1:, :]
    return utils.rowstochastic(counts)


def get_membership(timeseries, centers, sigma):
    """ Compute the pairwise membership / probability of each datapoint 
    to the Ansatz functions around each center. """
    sqdist = distance.cdist(timeseries, centers, distance.sqeuclidean)
    gausskernels = np.exp(-sqdist / (2*sigma**2))
    return utils.rowstochastic(gausskernels)
