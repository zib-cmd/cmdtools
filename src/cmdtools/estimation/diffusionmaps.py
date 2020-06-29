import scipy.linalg as lin
import scipy.spatial.distance as scidist
import numpy as np
import warnings
import sklearn.neighbors as neighbors
from cmdtools.utils import rowstochastic


class DiffusionMaps:

    def __init__(self, X, sigma=1, alpha=1, metric='sqeuclidean', n=1):
        self.X = X
        self.sigma = sigma
        self.alpha = alpha
        self.metric = metric
        self.n = n

        self.diffusion_matrix()
        self.diffusionmaps()

    def diffusion_matrix(self):
        self.P, self.q = diffusion_matrix(
            self.X, self.sigma, self.alpha, self.metric)

    def diffusionmaps(self):
        self.dms, self.evals, self.evecs = diffusionmaps(self.P, self.n)

    def oos_extension(self, Xnew):
        return oos_extension(
            Xnew, self.X, self.metric, self.sigma, self.alpha,
            self.q, self.dms, self.evals)


def diffusion_matrix(X, sigma, alpha, metric):
    # unnormalized kernel
    dist = scidist.squareform(scidist.pdist(X, metric))  # TODO: approximate with nearest neighbours
    sqd = dist if metric == 'sqeuclidean' else dist**2
    K = np.exp(-sqd / (2 * sigma**2))

    # pre normalization
    q = np.power(np.sum(K, axis=1), -alpha)
    Kh = np.diag(q).dot(K).dot(np.diag(q))

    # row-normalization
    P = rowstochastic(Kh)
    return P, q


def diffusionmaps(P, n):
    evals, evecs = lin.eig(P)  # TODO: check/compare with left eigenvectors
    idx = evals.argsort()[::-1][1:n+1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    dms = evals[None, :] * evecs
    if not np.all(np.isreal(dms)):
        warnings.warn("Diffusion map is not real")
    dms = np.real(dms)
    return dms, evals, evecs


def oos_extension(Xnew, Xold, metric, sigma, alpha, q, dms, evals):
    dist = scidist.cdist(Xnew, Xold, metric)
    sqd = dist if metric == 'sqeuclidean' else dist**2
    Knew = np.exp(-sqd / (2 * sigma**2))

    qnew = np.power(np.sum(Knew, axis=1), -alpha)

    Kh = np.diag(qnew).dot(Knew).dot(np.diag(q))
    Pnew = rowstochastic(Kh)

    dmsnew = Pnew.dot(dms) / evals[None, :]
    return dmsnew


def test_diffusionmaps():
    X = np.random.rand(100, 100)
    dm = DiffusionMaps(X)
    assert np.allclose(dm.oos_extension(X), dm.dms)


def nearest_neightbor_dists(X, k, metric):
    return neighbors.kneighbors_graph(X, k, mode='distance', metric=metric)


def superdist(X, Y = None, metric='squeclidean', squared=True, knn=None):
    if Y is X:
        Y = None
    if knn not None:
        if Y not None:
            raise Error("Nearest neighbor not defined for X neq Y")
        d = neighbors.kneighbors_graph(X, knn, mode='distance', metric=metric)
    else:
        if Y is None:
            d = scidist.squareform(scidist.pdist(X, metric))
        else:
            d = scidist.cdist(X, Y, metric)

    if squared and metric is not 'squeclidean':
        d = d**2

    return d