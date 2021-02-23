from cmdtools import utils
from cmdtools.estimation import sqra
import numpy as np


def test_sqra():
    grids = 30  # number of intervals in [0,1]-space (e.g. 30)
    beta  = 10  # inverse temperature for Boltzmann (e.g. 10)

    # grid & flux computation according to Luca Donati
    grid = np.linspace(0, 1, grids+1)
    phi = grids**2 / beta / 9

    # potential
    u = 50*(grid-0.6)**2

    # adjacency matrix of the intervals for Q-assembly
    A = np.diag(np.ones(grids), k=1)
    A = A + A.T

    return sqra.sqra(u, A, beta, phi)


def test_SQRA():
    u = np.random.rand(10)
    sqra.SQRA(u)
