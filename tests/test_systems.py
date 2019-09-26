from cmdtools.systems import diffusion


def test_diffusion():
    diffusion.DiffusionProcess(lambda x: x)
    diffusion.DiffusionProcess(lambda x: x, epsilon=1)


def test_doublewell():
    diffusion.DoubleWell()


def test_threehole():
    d = diffusion.ThreeHolePotential()
    d.beta = 1.6
    d.stationary_dist_nn([0, 0])
