
import numpy as np
import scipy.spatial.distance as dist


def picking_algorithm(X, n):
    """Picking algorithm (Durmaz, 2016)

    Pick out n points such that the respective minimal distance
    to the other picked points is maximized.

    Parameters
    ----------
    X : (N x d) matrix
        Matrix of N d-dimensional points to pick from
    n : int
        Number of points to pick.
    Returns
    -------
    P : (n x d) matrix
        Matrix containing the n chosen points
    qs : list[int]
        List of the original indices of the chosen points.
    d : (n x n) matrix
        Distances between the chosen points
    """
    d = np.empty((n, np.size(X, 0)))
    d[0, :] = dist.cdist(X[[0], :], X)
    qs = [0]
    for k in range(1, n):
        q = np.argmax(np.min(d[:k, :], axis=0))
        d[k, :] = dist.cdist(X[[q], :], X)
        qs.append(q)
    return X[qs, :], qs, d[:, qs]
