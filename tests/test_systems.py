from cmdtools.systems import diffusion


def test_doublewell():
    diffusion.DoubleWell()


def test_tripplewell():
    d = diffusion.TripleWell()
    d.Q
