import scipy.linalg as lin
import scipy.sparse as spr
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
            d = d.toarray()
            # symmetrize
            d = d + (d == 0)*d.T
        else:
            n = neighbors.NearestNeighbors(metric=self.metric)
            n.fit(Y)
            d = n.kneighbors_graph(X, self.k, mode='distance')
            d = d.toarray()
        return d


class DiffusionMaps:
    def __init__(self, X, sigma=1, alpha=1, distances=Distances(), n=1):
        self.X = X
        self.alpha = alpha
        self.distances = distances
        self.n = n
        # unnormalized kernel
        self.sqd = distances.sqdist(self.X)

        if isinstance(sigma, np.ndarray):
            self.sigma = self.bandwidth_estimator(range_exp=sigma)
        elif isinstance(sigma, int) or isinstance(sigma, float):
            self.sigma = sigma
        elif sigma == 'estimate':
            self.sigma = self.bandwidth_estimator()
        else:
            print('Did not recognize the given sigma.')

        self.diffusion_matrix()
        self.diffusionmaps()

    def diffusion_matrix(self):
        self.P, self.q = diffusion_matrix(
            self.sqd, self.sigma, self.alpha)

    def diffusionmaps(self):
        self.dms, self.evals, self.evecs = diffusionmaps(self.P, self.n)

    def oos_extension(self, Xnew):
        return oos_extension(
            Xnew, self.X, self.distances, self.sigma, self.alpha,
            self.q, self.dms, self.evals)

    def bandwidth_estimator(self, range_exp=np.arange(-20., 10.)):
        return bandwidth_estimator(
            self.sqd, range_exp)


def diffusion_matrix(sqd, sigma, alpha):

    K = np.exp(-sqd / (2 * sigma**2))

    # pre normalization
    q = np.power(np.sum(K, axis=1), -alpha)
    Kh = np.diag(q).dot(K).dot(np.diag(q))

    # row-normalization
    P = rowstochastic(Kh)
    return P, q


def diffusionmaps(P, n):
    # P=spr.csc_matrix(P)
    evals, evecs = spr.linalg.eigs(P, n+1, which='LR')  # largest real value
    idx = evals.argsort()[::-1][1:]
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


def bandwidth_estimator(sqd, range_exp):
    '''
    Bandwidth estimator searching for the optimal sigma in 2^range_exp
    according to the heuristics from:
    Berry, Tyrus, and John Harlim. "Variable bandwidth diffusion kernels."
    Applied and Computational Harmonic Analysis 40.1 (2016): 68-96.
    '''
    sigmas = 2**range_exp  # range of sigmas

    log_S = np.array([np.log(np.sum(1/np.size(sqd)
                             * np.exp(-sqd / (2 * s**2)))) for s in sigmas])
    log_sig = np.log(sigmas**2)
    index = np.argmax((log_S[1:]-log_S[:-1])/(log_sig[1:]-log_sig[:-1]))

    sigma = sigmas[index]
    return sigma
