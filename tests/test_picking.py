import cmdtools.estimation.picking_algorithm as pick
import numpy as np


def test_bench_prealloc(benchmark, n=100, m=10, d=3):
    X = np.random.rand(n, d)
    benchmark(pick.picking_prealloc, X, m)


def test_bench_fast(benchmark, n=100, m=10, d=3):
    X = np.random.rand(n, d)
    benchmark(pick.picking_fast, X, m)


def test_bench_old(benchmark, n=100, m=10, d=3):
    X = np.random.rand(n, d)
    benchmark(pick.picking_algorithm, X, m)

# n=100,  m=100, d=2: 20x faster (worst case)
# n=1000, m=100, d=2: 242x
# n=1000, m=35,  d=2: 3000x (ratio of diala, but 10x lower n => expect 30000x)

# old algorithm is O(n^2)
# new one is O(n*m)
# and has a better constant since were not dealing with the intermediate list


def test_comparepick(n=100, m=10, d=3):
    X = np.random.rand(n, d)
    _, p1 = pick.picking_algorithm(X, m)
    p2 = pick.picking_fast(X, m)
    p3 = pick.picking_prealloc(X, m)
    assert np.all(np.isclose(p1, p2))
    assert np.all(np.isclose(p1, p3))
