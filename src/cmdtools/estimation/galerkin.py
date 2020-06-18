from .. import utils
import numpy as np
from scipy.spatial import distance


class Gaussian:
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
        self.mass = massmatrix(self.membership)

    @property
    def propagator(self):
        return propagator(self.membership)


def massmatrix(membership):
    """ Compute the row-stochastic empirical mass function
    S_ij = St_ij / sum_j St_ij
    St_ij = 1/n sum_x phi_i(x) phi_j(x), x in traj
    cf. ([TGDW2019] eqn 12, [W2006] eqn 35)

    Args:
        membership (matrix): row-stochastic membership matrix
            of shape (#samples x #basis)

    Returns:
        matrix: row-stochastic mass matrix S
    """

    St = membership.T.dot(membership)
    return utils.rowstochastic(St)


# TODO: isn't this actually the koopman operator?
def propagator(membership, lag=1):
    """Compute the row-stochastic empirical propagator
    K_ij = Kt_ij / sum_j Kt_ij
    Kt_ij = 1/n sum_x phi_i(x) phi_j(y), (x,y) in traj
    cf. ([TGDW2019] eqn 12, [W2006] eqn 35)

    Args:
        membership (matrix): row-stochastic membership matrix
            of shape (#samples x #basis)

    Returns:
        matrix: row-stochastic propagator P
    """

    counts = membership[0:-lag, :].T.dot(membership[lag:, :])
    p = utils.rowstochastic(counts)
    return p


def membership(sqd, sigma):
    """
    Given square distances and standard deviation,
    return the row-stochastic Gaussian membership matrix
    """
    m = np.exp(-sqd / (2 * sigma**2))
    return utils.rowstochastic(m)


def sqdist(timeseries, centers, metric='sqeuclidean'):
    d = distance.cdist(timeseries, centers, metric)
    sqd = d if metric == 'sqeuclidean' else d**2
    return sqd


def find_bandwidth(sqd, percentile=50):
    """

    Given the square-distance matrix d_ij = (x_i - x_j)^2
    compute the standard deviation s for the Gaussian kernel
    k(x_i,x_j) = exp(-1/(2s^2) d_ij).
    The heuristic is based on the assumption that we want
    sum_j k(x_i, x_j) ≈ n exp(-1/(2s^2) med^2) = 1 and hence
    (2s^2) = med^2 / log n  where n is the number of points and
    med the pairwise median distance.

    percentile allows to shift from the median to an arbitrary percentage and
    thus influences how many points have an influence on the bandwith.

    Heuristic taken from "Stein Variational Gradient Descent [...],
    Qiang Liu and Dilin Wang (2016)".

    Input:
        sqd: matrix, pairwise squared-distances of the samples
        percentile: float [0,100], percentile to use instead of median

     Output:
         sigma: float, the standard deviation of the Gaussian
    """

    n = np.size(sqd, 1)
    h = np.percentile(sqd, percentile) / np.log(n)
    sigma = np.sqrt(h/2)
    return sigma
