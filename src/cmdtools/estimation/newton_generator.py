import numpy as np
import copy
from .galerkin import propagator
from math import factorial


def lagged_propagators(membership, mass, maxlag=1):
    n = np.size(membership, 1)
    props = np.zeros((maxlag+1, n, n))
    props[0, :, :] = mass
    for i in range(1, maxlag + 1):
        props[i, :, :] = propagator(membership, i)
    return props


def newton_generator(traj, order):
    props = lagged_propagators(traj.membership, traj.mass, maxlag=order)
    q = Newton_N(props, 1, 0)
    return q


def div_diff(P):
    """Function computing the divided difference formula for interpolation points
    with equal discretization distance (later called h). Return the divided differences
    for the given matrices in the tensor P, but without binomial coefficient in from of
    each result.
    Input:
        P= tensor
    Output:
        tensor containing the divided differences values."""
    t = int(np.shape(P)[0])
    n = int(np.shape(P)[1])
    D = np.zeros((t, t, n, n))
    D[:, 0, :, :] = copy.deepcopy(P)
    d = 1
    for j in range(t-1):
        for i in range(j+1, t):

            D[i, j+1, :, :] = (D[i, j, :, :]-D[i-1, j, :, :])

            d += 1

    # ,D
    return np.transpose(np.reshape(np.transpose(D.diagonal(0, 0, 1)), (t, n, n)), axes=(0, 2, 1))


def Newton_N(B, h, x):
    """Function for computing the transition rate matrix at the point x from the
    time series of the transition probability matrix (contained in the vector B)
    with the discretization contant h.
    The method used is the Newton's polynomial extrapolation derived at the
    point x.    The programmed formula is the one in paragraph 13.2 of
    "Formelsammlung for numerische Mathematik in C-Programming",
    Engeln-Muellges, Reutter, SI Wissenschaftsverlag, 1990.


    Input:
    B=tensor
    h=float,step size
    x=int,point of the extrapolation
    Output:
        matrix representing the values of the polynomial at the point x(2d-
        array)"""
    Deltas = div_diff(B)
    n_prime = int(np.shape(B)[0])
    x = int(x)
    big_pi = 1.
    sigma_fin = 0.
    big_sigma = 0.

    for j in range(n_prime):
        big_sigma = 0.

        for k in range(0, j):
            # print("k",k,"j")
            big_pi = 1.
            for i in range(0, j):
                # print
                if i == k:
                    continue
                else:
                    big_pi *= (x-i)
            big_sigma += big_pi

        sigma_fin += big_sigma*Deltas[j]/np.float32(factorial(j))  # TODO: fix this adhoc cast

    return(sigma_fin/h)


def Newton2(B, h, x):
    """Make sure that the sum of the entries of each row of the generator
    matrix is zero. To do it, it apply the function Newton_N and then make the
    sum of each row equal zero.

    Input:
    B=tensor
    h= float,step size
    x=point of the interpolation
    Output:
        matrix representing the values of the polynomial at the point x(2d-
        array)
        """
    diff = copy.deepcopy(Newton_N(B, h, x))
    for j in range(diff.shape[1]):
        diff[j, j] = -(np.sum(diff[j, :])-diff[j, j])
    return(diff)
