from cmdtools.systems import diffusion


def test_diffusion():
    diffusion.DiffusionProcess(lambda x: x)
    diffusion.DiffusionProcess(lambda x: x, epsilon=1)


def test_doublewell():
    diffusion.DoubleWell()


def test_threehole():
    diffusion.ThreeHolePotential()
