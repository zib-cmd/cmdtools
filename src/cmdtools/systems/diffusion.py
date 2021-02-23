import numpy as np
from ..estimation import sqra


class AbstractDiffusion:
    def __init__(self, nx, ny, xlims, ylims, beta=1, phi=1):

        xs = np.linspace(*xlims, nx)
        ys = np.linspace(*ylims, ny)
        self.xs, self.ys = np.meshgrid(xs, ys, sparse=True)
        self.beta = beta
        self.phi = phi

        self.u = self.potential()
        self.sqra = sqra.SQRA(self.u, beta=self.beta, phi=self.phi)
        self.Q = self.sqra.Q


class DoubleWell(AbstractDiffusion):
    def __init__(self, nx=5, ny=1, xlims=(-2.5, 2.5), ylims=(-1, 1), **kwargs):
        super().__init__(nx=nx, ny=ny, xlims=xlims, ylims=ylims, **kwargs)

    def potential(self):
        return (self.xs**2-1)**2 + self.ys**2


class TripleWell(AbstractDiffusion):
    def __init__(self, nx=4, ny=3, xlims=(-2, 2), ylims=(-1, 2), **kwargs):
        super().__init__(nx=nx, ny=ny, xlims=xlims, ylims=ylims, **kwargs)

    def potential(self):
        x = self.xs
        y = self.ys
        V = (3/4 * np.exp(-x**2 - (y-1/3)**2)
            - 3/4 * np.exp(-x**2 - (y-5/3)**2)
            - 5/4 * np.exp(-(x-1)**2 - y**2)
            - 5/4 * np.exp(-(x+1)**2 - y**2)
            + 1/20 * x**4 + 1/20 * (y-1/3)**4)
        return V
