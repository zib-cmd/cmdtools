import numpy as np
from cmdtools.estimation.newton_generator import Newton_N


def test_Newton():

    tensor = np.random.randint(0, 10, size=(4, 4, 4))
    diff_tensor = np.zeros((4, 4, 4))

    diff_tensor[0, :, :] = 1/(6.) * (-11*tensor[0, :, :] + 18*tensor[1, :, :]
                                     - 9*tensor[2, :, :] + 2*tensor[3, :, :])

    diff_tensor[1, :, :] = 1/(6.) * (-2*tensor[(0), :, :] - 3*tensor[1, :, :]
                                     + 6*tensor[2, :, :] - tensor[3, :, :])

    diff_tensor[2, :, :] = 1/(6.) * (tensor[0, :, :]-6 * tensor[1, :, :]
                                     + 3*tensor[2, :, :] + 2*tensor[3, :, :])

    diff_tensor[3, :, :] = 1/(6.) * (-2*tensor[0, :, :] + 9*tensor[1, :, :]
                                     - 18*tensor[2, :, :] + 11*tensor[3, :, :])

    applied_Newton = np.zeros((4, 4, 4))

    for i in range(4):

        applied_Newton[i] = Newton_N(tensor, 1., i)

    assert np.alltrue(np.round(diff_tensor, 5) == np.round(applied_Newton, 5))
