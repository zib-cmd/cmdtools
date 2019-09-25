import numpy as np


class DiffusionProcess:
    """ Representation of the diffusion process
    dX = ∇U(X) dt + √ε dW """
    def __init__(self, U, epsilon):
        self.U = U
        self.epsilon = epsilon

    def stationary_dist_nn(self, x):
        return np.exp(-self.U(x))


class DoubleWell(DiffusionProcess):
    def __init__(self):
        def U(x):
            return (x**2 - 1)**2 / 4
        super().__init__(U, 0.03)

    xmin = -2.5
    xmax = 2.5


class ThreeHolePotential(DiffusionProcess):
    """ The Three-Hole Potential from 2006 - Metzner, Schütte, Vanden-Eijnden -
    Illustration of transition path theory """
    def __init__(self, beta=1.67):
        def potential(xy):
            x = xy[0]
            y = xy[1]
            return 3 * np.exp(-x**2 - (y-1/3)**2) \
                - 3 * np.exp(-x**2 - (y-5/3)**2) \
                - 5 * np.exp(-(x-1)**2 - y**2) \
                - 5 * np.exp(-(x+1)**2 - y**2) \
                + .2 * x**4 + .2 * (y-1/3)**4

        super().__init__(potential, 2/beta)

    xmin = -2
    xmax = 2
    ymin = -1.5
    ymax = 2.5
