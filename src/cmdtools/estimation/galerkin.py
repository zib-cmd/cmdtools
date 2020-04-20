from .. import utils
import numpy as np
from scipy.spatial import distance


def propagator(timeseries, centers, sigma):
    """ Given `timeseries` data, estimate the propagator matrix.

    Uses the galerkin projection onto Gaussian ansatz functions
    with bandwidth `sigma` around the given `centers`. """
    m = get_membership(timeseries, centers, sigma)
    counts = m[0:-1, :].T.dot(m[1:, :])
    return utils.rowstochastic(counts)


def get_membership(timeseries, centers, sigma):
    """ Compute the pairwise membership / probability of each datapoint
    to the Ansatz functions around each center. """
    sqdist = distance.cdist(timeseries, centers, distance.sqeuclidean)
    gausskernels = np.exp(-sqdist / (2 * sigma**2))
    return utils.rowstochastic(gausskernels)


def find_bandwidth(timeseries, centers, percentile=50):  # , plot= True):
    """Find the bandwidth of the Gaussian based on: 

    "Stein Variational Gradient Descent: 
    A General Purpose Bayesian Inference Algorithm", 
    Qiang Liu and Dilin Wang (2016).

     Based on the value of the percentile is possible to decide the points to
     take into consideration for the determination of the bandwidth.

    Input:
        timeseries: arr, trajectory, each row is a collection of coordinates at a
            different timestep
        centers: arr, centers of the Gaussians, each row has the coordinates 
            of a different center
        percentile: int [0,100], default value = 50

     Output:
         sigma: float, the  variance of the Gaussian"""

    no_centers = np.shape(centers)[0]
    sqdist = distance.cdist(timeseries, centers, distance.sqeuclidean)


#   uncomment to plot
#    if plot == True:
#        plt.hist(sqdist.flatten())
#        plt.vlines(np.percentile(sqdist,percentile), 0, 10, label= percentile)
#        plt.grid()
#        plt.title("Plot of the histogram with Euclidean distances and chosen percentile")
#        plt.legend()
#        plt.show()

# since we have h = perc**2/log(n) = 2 * sigma**2
    return np.percentile(sqdist, percentile) / np.sqrt(2*np.log(no_centers))
