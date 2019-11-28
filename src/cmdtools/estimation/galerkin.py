from .. import utils
import numpy as np
from scipy.spatial import distance


class Trajectory:
    def __init__(self, timeseries, centers=None, sqd=None, sigma=None, percentile=50):
        self.timeseries = timeseries
        self.centers = timeseries if centers is None else centers
        self.sqd = sqdist(
            self.timeseries, self.centers) if sqd is None else sqd
        self.sigma = find_bandwidth(
            self.sqd, percentile) if sigma is None else sigma

    @property
    def membership(self):
        return membership(self.sqd, self.sigma)

    @property
    def propagator(self):
        return propagator(self.membership)


def lagged_propagator(m, lag):
    p = np.zeros([m.shape[1], m.shape[1]])
    for i in range(0,lag):
        p += propagator(m[i::lag,:])
    return p / lag


def propagator(m):
    " given the membership of a trajectory to some basis functions, estimate the propagator "
    counts = m[0:-1, :].T.dot(m[1:, :])
    return utils.rowstochastic(counts)


def membership(sqd, sigma):
    m = np.exp(sqd / (2 * sigma**2))
    return utils.rowstochastic(m)


def sqdist(timeseries, centers):
    return distance.cdist(timeseries, centers, distance.sqeuclidean)


def find_bandwidth(sqd, percentile=50):
    """Find the bandwidth of the Gaussian based on: 

    "Stein Variational Gradient Descent: 
    A General Purpose Bayesian Inference Algorithm", 
    Qiang Liu and Dilin Wang (2016).

     Based on the value of the percentile is possible to decide the points to
     take into consideration for the determination of the bandwidth.

    Input:
        timeseries: arr, trajectory, each row is a collection of 
            coordinates at a different timestep
        centers: arr, centers of the Gaussians, each row has the coordinates 
            of a different center
        percentile: int [0,100], default value = 50

     Output:
         sigma: float, the  variance of the Gaussian"""

    no_centers = np.shape(sqd)[1]

    # since we have h = perc**2/log(n) = 2 * sigma**2
    return np.percentile(sqd, percentile) / np.sqrt(2*np.log(no_centers))
