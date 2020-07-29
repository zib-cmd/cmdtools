from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
from .. import utils
from .picking_algorithm import picking_algorithm


class VoronoiTrajectory:
    def __init__(self, traj, centers='kmeans', n=1):
        self.traj = traj
        if centers == 'kmeans':
            self.centers, self.inds = by_kmeans(traj, n)
        elif centers == 'picking':
            self.centers, self.inds = by_picking(traj, n)
        else:
            self.centers, self.inds = by_nn(traj, centers)

    def propagator(self, dt=1):
        return propagator(self.inds, np.size(self.centers, 0), dt)


def propagator(inds, nstates, dt):
    P = np.zeros((nstates, nstates))
    for i in range(len(inds)-dt):
        P[inds[i], inds[i+dt]] += 1
    return utils.rowstochastic(P)


def by_nn(X, centers):
    inds = (NearestNeighbors()
            .fit(centers).kneighbors(X, 1, False)
            .reshape(-1))
    return centers, inds


def by_kmeans(X, n):
    k = KMeans(n_clusters=n).fit(X)
    inds = k.labels_
    centers = k.cluster_centers_
    return centers, inds


def by_picking(X, n):
    centers, _, d = picking_algorithm(X, n)
    inds = np.argmax(d, axis=1)
    return centers, inds


class GridTrajectory:
    def __init__(self, traj, lims, ns):
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

        bti = boxtrajinds(traj, lims, ns)
        b, ti = np.unique(bti, return_inverse=True)
        # b[ti] == bti

        self.boxinds = b
        self.centers = boxcenters(b, lims, ns)
        self.traj = ti

    def propagator(self, dt=1):
        return propagator(self.traj, len(self.boxinds), dt)

def getboxes(traj, lims=None, ns=1):
    if traj.ndim == 1:
        traj = traj.reshape(np.size(traj), 1)
    if lims is None:
        mins = np.min(traj, 0)
        maxs = np.max(traj, 0)
        lims = np.vstack((mins, maxs)).T
    else:
        lims = np.array(lims)
    if np.isscalar(ns):
        ns = np.repeat(ns, np.size(traj, 1))
    else:
        ns = np.array(ns)

def boxtrajinds(traj, lims=None, ns=1):
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
    boxes = boxes.astype(np.int)

    inds = np.ravel_multi_index(boxes.T, ns)
    return inds


def boxcenters(inds, lims, ns):
    lims = np.array(lims)
    ns   = np.array(ns)
    coords = np.empty((len(inds), len(ns)))
    scale = lims[:, 1] - lims[:, 0]
    unrav = np.vstack(np.unravel_index(inds, ns))
    for i in range(len(inds)):
        coords[i, :] = lims[:, 0] + (1/ns) * (unrav[:, i] + 1/2) * scale
    return coords


inds = [0, 1, 2]
lims = [[0, 1]]
ns = [3]

