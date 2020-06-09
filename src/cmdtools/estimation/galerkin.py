from .. import utils
import numpy as np
from scipy.spatial import distance


class Trajectory:
    def __init__(self, timeseries,
                 centers=None, sqd=None, sigma=None, percentile=50):
        self.timeseries = timeseries
        self.centers = timeseries \
            if centers is None else centers
        self.sqd = sqdist(self.timeseries, self.centers) \
            if sqd is None else sqd
        self.sigma = find_bandwidth(self.sqd, percentile) \
            if sigma is None else sigma

        self.membership = membership(self.sqd, self.sigma)
        self.mass = massmatrix(self.centers, self.sigma)

    @property
    def propagator(self):
        return propagator(self.membership, self.mass)


def massmatrix(centers, sigma):
    """ compute the inverse of the mass matrix / the overlap
    used to factor out the self-transitions induced by the overlapping basis"""
    m = membership(sqdist(centers, centers), sigma)
    p = m.T.dot(m)
    return p


def lagged_propagator(m, lag, mass=None):
    """
    Given the membership of a trajectory to some basis functions,
    estimate the propagator for a given time-lag.
    This is achieved by skipping the corresponding lag in the trajectory
    and averaging over the resulting subtrajectories.
    """
    p = np.zeros([m.shape[1], m.shape[1]])
    for i in range(0, lag):
        p += propagator(m[i::lag, :]) / lag
    p = p / lag
    if mass is not None:
        p = mass.dot(p)  # TODO: is this correct? see below
    return p


def propagator(m, mass=None):
    """
    Given the membership matrix of a trajectory to some basis functions
    estimate the propagator
    """
    counts = m[0:-1, :].T.dot(m[1:, :])
    p = counts
    if mass is not None:
        p = p.dot(np.linalg.inv(mass))  # TODO: is this correct? see above
    p = utils.rowstochastic(counts)
    return p


def membership(sqd, sigma):
    """
    Given square distances and standard deviation,
    return the row-stochastic Gaussian membership matrix
    """
    m = np.exp(-sqd / (2 * sigma**2))
    return utils.rowstochastic(m)


def sqdist(timeseries, centers):
    return distance.cdist(timeseries, centers, distance.sqeuclidean)


def find_bandwidth(sqd, percentile=50):
    """

    Given the square-distance matrix d_ij = (x_i - x_j)^2
    compute the standard deviation s for a Gaussian kernel
    k(x_i,x_j) = exp(-1/h d_ij) with bandwidth parameter h = 2s^2.
    We have h = med^2 / log n based on the intuition that we want
    sum_j k(x_i, x_j) ≈ n exp(-1/h med^2) = 1
    where n is the number of points and med the pairwise median distance

    Percentile allows to shift from the median to an arbitrary percentage and
    thus influences how many points have an influence on the bandwith.

    Reference:
    "Stein Variational Gradient Descent:
    A General Purpose Bayesian Inference Algorithm",
    Qiang Liu and Dilin Wang (2016).

    Input:
        sqd: matrix, pairwise squared-distances of the samples
        percentile: int [0,100], amount of samples to be in the "horizon" of h

     Output:
         sigma: float, the standard deviation of the Gaussian
    """

    n = np.size(sqd, 1)
    h = np.percentile(sqd, percentile)**2 / np.log(n)
    sigma = np.sqrt(h/2)
    return sigma
