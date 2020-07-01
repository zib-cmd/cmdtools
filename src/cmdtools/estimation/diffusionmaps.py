import scipy.linalg as lin
import scipy.spatial.distance as scidist
import numpy as np
import warnings
import sklearn.neighbors as neighbors
from cmdtools.utils import rowstochastic


# TODO: we also have distances in picking/galerkin - unite the code
class Distances:
    def __init__(self, metric='sqeuclidean'):
        self.metric = metric

    def dist(self, X, Y=None):
        if Y is X or Y is None:
            d = scidist.pdist(X, self.metric)
            return scidist.squareform(d)
        else:
            return scidist.cdist(X, Y, self.metric)

    def sqdist(self, X, Y=None):
        d = self.dist(X, Y)
        d = d ** 2 if self.metric != 'squeclidean' else d
        return d


class NNDistances(Distances):
    def __init__(self, k=1, *args, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def dist(self, X, Y=None):
        if Y is X or Y is None:
            d = neighbors.kneighbors_graph(
                X, self.k, mode='distance', metric=self.metric)
        else:
            n = neighbors.NearestNeighbors(metric=self.metric)
            n.fit(Y)
            d = n.kneighbors_graph(X, self.k, mode='distance')

        return d.toarray()  # since we cant deal with sparse so far


class DiffusionMaps:
    def __init__(self, X, sigma=1, alpha=1, distances=Distances(), n=1):
        self.X = X
        self.sigma = sigma
        self.alpha = alpha
        self.distances = distances
        self.n = n

        self.diffusion_matrix()
        self.diffusionmaps()

    def diffusion_matrix(self):
        self.P, self.q = diffusion_matrix(
            self.X, self.sigma, self.alpha, self.distances)

    def diffusionmaps(self):
        self.dms, self.evals, self.evecs = diffusionmaps(self.P, self.n)

    def oos_extension(self, Xnew):
        return oos_extension(
            Xnew, self.X, self.distances, self.sigma, self.alpha,
            self.q, self.dms, self.evals)


def diffusion_matrix(X, sigma, alpha, distances):
    # unnormalized kernel
    sqd = distances.sqdist(X)
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


def oos_extension(Xnew, Xold, distances, sigma, alpha, q, dms, evals):
    sqd = distances.sqdist(Xnew, Xold)

    Knew = np.exp(-sqd / (2 * sigma**2))
    qnew = np.power(np.sum(Knew, axis=1), -alpha)
    Kh = np.diag(qnew).dot(Knew).dot(np.diag(q))
    Pnew = rowstochastic(Kh)

    dmsnew = Pnew.dot(dms) / evals[None, :]
    dmsnew = np.real(dmsnew)
    return dmsnew


def test_diffusionmaps():
    X = np.random.rand(100, 5)
    Y = np.random.rand(10, 5)
    dm1 = DiffusionMaps(X)
    dm2 = DiffusionMaps(X, distances=NNDistances(20))
    for dm in [dm1, dm2]:
        assert np.allclose(dm.oos_extension(X), dm.dms)
        dm.oos_extension(Y)