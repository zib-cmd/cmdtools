from cmdtools.systems import diffusion
from cmdtools.estimation import ajcs
import numpy as np
from scipy.linalg import expm


def test_ajc(n=20, tol=1e-2):
    Q = diffusion.DoubleWell().Q
    ts = np.repeat(1/n, n)
    a = ajcs.AJCS(np.repeat(Q, n), ts)

    assert np.isclose(expm(Q).todense(), a.koopman(), atol=tol).all()

    g = np.zeros((a.nt, a.nx))
    g[0, :] = np.nan
    g[1, 1] = 1
    a.space_time_committor(g)


def test_limit(tol=1e-5):
    Q = diffusion.DoubleWell().Q
    a0 = ajcs.AJCS([Q, Q, Q], [0,1,0])
    a1 = ajcs.AJCS([Q, Q, Q], [1e-10, 1, 1e-10])

    assert np.isclose(a1.koopman(), a0.koopman(), atol=tol).all()
