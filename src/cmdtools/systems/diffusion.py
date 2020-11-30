import numpy as np
from ..estimation import sqra


class DiffusionProcess:
    """ Representation of the diffusion process
    dX = -∇V(X) dt + √(2ε) dW """  # TODO: check if sqrt is right here
    def __init__(self, V, epsilon=1):
        self.V = V
        self.epsilon = epsilon

    def stationary_dist_nn(self, x):
        """ Evaluate the non-normalized Boltzmann distribution at x """
        return np.exp(-self.beta * self.V(x))

    @property
    def beta(self):
        return 2 / self.epsilon

    @beta.setter
    def beta(self, b):
        self.epsilon = beta_to_epsilon(b)


class DoubleWell(DiffusionProcess):
    def __init__(self):
        def potential(x):
            return (x**2 - 1)**2 / 4
        DiffusionProcess.__init__(self, potential, beta_to_epsilon(1))

    xmin = -2.5
    xmax = 2.5
    dimension = 1


class ThreeHolePotential(DiffusionProcess):
    """ The Three-Hole Potential from 2006 - Metzner, Schütte, Vanden-Eijnden -
    Illustration of transition path theory """
    def __init__(self):
        def potential(xy):
            x = xy[0]
            y = xy[1]
            return 3 * np.exp(-x**2 - (y-1/3)**2) \
                - 3 * np.exp(-x**2 - (y-5/3)**2) \
                - 5 * np.exp(-(x-1)**2 - y**2) \
                - 5 * np.exp(-(x+1)**2 - y**2) \
                + .2 * x**4 + .2 * (y-1/3)**4

        DiffusionProcess.__init__(self, potential, beta_to_epsilon(1.67))

    dimension = 2
    xmin = -2
    xmax = 2
    ymin = -1.5
    ymax = 2.5

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