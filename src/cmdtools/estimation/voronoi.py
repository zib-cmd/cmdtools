from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from warnings import warn
import numpy as np
from .. import utils
from .picking_algorithm import picking_algorithm


class VoronoiTrajectory:
    def __init__(self, traj, n, centers='kmeans', metric='sqeuclidean'):
        self.traj = traj
        self.metric = metric
        if not np.isscalar(centers):  # we pass an array
            self.centers, self.inds = by_nn(traj, centers, metric)
        elif centers == 'kmeans':
            self.centers, self.inds = by_kmeans(traj, n, metric)
        elif centers == 'picking':
            self.centers, self.inds = by_picking(traj, n, metric)
        else:
            raise ValueError("invalid centers")

    def propagator(self, dt=1):
        return propagator(self.inds, np.size(self.centers, 0), dt)


def propagator(inds, nstates, dt):
    P = np.zeros((nstates, nstates))
    for i in range(len(inds)-dt):
        P[inds[i], inds[i+dt]] += 1
    P = utils.rowstochastic(P)
    if not utils.is_rowstochastic(P):
        warn("Estimated propagator is not row-stochastic. Possibly ergodicity is not satisfied, consider more samples or a coarser resolution")
    return utils.rowstochastic(P)


def by_nn(X, centers, metric='sqeuclidean'):
    inds = (NearestNeighbors(metric=metric)
            .fit(centers).kneighbors(X, 1, False)
            .reshape(-1))
    return centers, inds


def by_kmeans(X, n, metric='sqeuclidean'):
    if metric not in ["sqeuclidean", "euclidean"]:
        raise ValueError("KMeans only supports (sq)euclidean metric")
    k = KMeans(n_clusters=n).fit(X)
    inds = k.labels_
    centers = k.cluster_centers_
    return centers, inds


def by_picking(X, n, metric='sqeuclidean'):
    centers, _, d = picking_algorithm(X, n, metric)
    inds = np.argmin(d, axis=1)
    return centers, inds


class SparseBoxes:
    """ Sparse box representation of a trajectory.
    Given a trajectory of n points in m-dimensional space,
    assign its points to the boxes of a regular grid and compute the associated transfer operator.
    The trajectory is represented in a sparse form, only storing boxes which are visited.

    Args:
        traj (ndarray): (n x m) array containing the sequential points of the trajectory
        lims (ndarray): (m x 2) array containing the minima and maxima of of the grid in the respective m dimensions
        ns (scalar, ndarray): number of boxes in each respective (or all) dimensions
    """

    def __init__(self, traj, lims=None, ns=1):
        if traj.ndim == 1:
            traj = traj.reshape(np.size(traj), 1)
        if lims is None:
            mins = np.min(traj, 0)
            maxs = np.max(traj, 0)
            lims = np.vstack((mins, maxs)).T
        else:
            lims = np.array(lims)
        self.lims = lims

        if np.isscalar(ns):
            ns = np.repeat(ns, np.size(traj, 1))
        else:
            ns = np.array(ns)
        self.ns = ns

        bti = boxtrajinds(traj, lims, ns)  # assignment of each traj. to the enumeration of dense cells
        b, ti = np.unique(bti, return_inverse=True)  # assignment to sparse cells
        # assert b[ti] == bti

        self.boxinds = b  # boxinds[i] contains the dense index of the j-th sparse cell
        self.centers = boxcenters(b, lims, ns)  # the coordinates corresponding to the sparse cells
        self.traj = ti  # trajectory in terms of the sparse cell enumeration

    def propagator(self, dt=1):
        return propagator(self.traj, len(self.boxinds), dt)


def boxtrajinds(traj, lims, ns):
    """ given n points in m-d space, return the indices of the dense boxes """
    scale = lims[:, 1] - lims[:, 0]
    normalized = (traj - lims[:, 0]) / scale
    n = np.size(traj, 0)
    boxes = np.empty((n, len(ns)), np.int32)

    # assign points to its box in each dimension
    for i in range(n):
        boxes[i, :] = normalized[i, :] * ns

    boxes = np.floor(boxes)

    # adjust for points laying at the boundary of the last box
    boxes = np.where(boxes == ns, ns-1, boxes)
    boxes = boxes.astype(int)

    inds = np.ravel_multi_index(boxes.T, ns)
    return inds


def boxcenters(inds, lims, ns):
    """ given the indices of the dense boxes, return their centers' coordinates """
    lims = np.array(lims)
    ns   = np.array(ns)
    coords = np.empty((len(inds), len(ns)))
    scale = lims[:, 1] - lims[:, 0]
    unrav = np.vstack(np.unravel_index(inds, ns))
    for i in range(len(inds)):
        coords[i, :] = lims[:, 0] + (1/ns) * (unrav[:, i] + 1/2) * scale
    return coords
