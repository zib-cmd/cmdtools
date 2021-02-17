from cmdtools.systems import diffusion
from cmdtools.estimation import ajcs
import numpy as np


def test_ajc():
    Q = diffusion.DoubleWell().Q
    a = ajcs.AJCS([Q, Q], [1, 1])

    g = np.zeros((a.nt, a.nx))
    g[0, :] = np.nan
    g[1, 1] = 1
    a.space_time_committor(g)


def test_jumpkernel2():
    Q = diffusion.DoubleWell().Q
    a = ajcs.AJCS([Q, Q], [1, 1])

    j1 = np.vstack([i.todense() for i in np.reshape(a.jumpkernel(a.qt, a.qi, a.dt)[0], 4)])
    j2 = np.vstack([i.todense() for i in np.reshape(a.jumpkernel2(a.qt, a.qi, a.dt)[0], 4)])
    assert(np.isclose(j1,j2).all())