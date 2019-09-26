import numpy as np


def beta_to_epsilon(b):
    return 2 / b


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
        super().__init__(lambda x: (x**2 - 1)**2 / 4, beta_to_epsilon(1))

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

        super().__init__(potential, beta_to_epsilon(1.67))

    dimension = 2
    xmin = -2
    xmax = 2
    ymin = -1.5
    ymax = 2.5
