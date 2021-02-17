from cmdtools.systems import diffusion
from cmdtools.estimation import ajcs
import numpy as np
from scipy.linalg import expm


def test_ajc():
    Q = diffusion.DoubleWell().Q
    a = ajcs.AJCS(np.repeat(Q, 10), np.repeat(0.1, 10))

    assert((expm(Q) - a.koopman() < .1).all())

    g = np.zeros((a.nt, a.nx))
    g[0, :] = np.nan
    g[1, 1] = 1
    a.space_time_committor(g)
