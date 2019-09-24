import numpy as np


def sqra(u, A, beta, phi):
    pi  = np.sqrt(np.exp(- beta * u))  # Boltzmann distribution
    pi /= np.sum(pi)

    D  = np.diag(pi)
    D1 = np.diag(1 / pi)
    Q  = phi * D1 @ A @ D
    return Q
