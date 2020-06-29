import scipy.linalg as lin
import scipy.spatial.distance as scidist
import numpy as np



def diffusion_matrix(X, sigma, alpha, metric):
    # unnormalized kernel
    dist = scidist.squareform(scidist.pdist(X, metric))  # TODO: approximate with nearest neighbours
    sqd = dist if metric == 'sqeuclidean' else dist**2
    K = np.exp(-sqd / (2 * sigma**2))

    # pre normalization
    q = np.power(np.sum(K, axis=1), -alpha)
    Kh = np.diag(q).dot(K).dot(np.diag(q))
    P = Kh / np.sum(Kh, axis=1)[:, None]
    return P, q


def diffusionmaps(P, n):
    evals, evecs = lin.eig(P)  # TODO: check/compare with left eigenvectors
    idx = evals.argsort()[::-1][1:n+1]
    dms = evals[None, idx] * evecs[:, idx]
    dms = np.real(dms)
    return dms, evals[idx], evecs[:, idx]


def oos_extension(Xnew, Xold, metric, sigma, alpha, q, dms, evals):
    dist = scidist.cdist(Xnew, Xold)
    sqd = dist if metric == 'sqeuclidean' else dist**2
    Knew = np.exp(-sqd / (2 * sigma**2))

    qnew = np.power(np.sum(Knew, axis=1), -alpha)

    Kh = np.diag(qnew).dot(Knew).dot(np.diag(q))
    Pnew = Kh / np.sum(Kh, axis=1)[:, None]

    dmsnew = Pnew.dot(dms) / evals[None, :]
    return dmsnew


def test_oos():
    X = np.random.rand(100, 100)
    sigma = 1
    alpha = 1
    metric = 'sqeuclidean'
    n = 10
    P, q = diffusion_matrix(X, sigma, alpha, metric)
    dms, evals, evecs = diffusionmaps(P, n)

    dmsnew = oos_extension(X, X, metric, sigma, alpha, q, dms, evals)

    assert np.allclose(dms, dmsnew)
