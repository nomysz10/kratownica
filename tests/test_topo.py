import numpy as np

from truss.mesh import make_rect_mesh
from truss.topo import density_filter, heaviside_projection


def test_filter_preserves_volume():
    rng = np.random.default_rng(0)
    m = make_rect_mesh(120.0, 40.0, 2.0)
    x = rng.random(m.elements.shape[0])
    y = density_filter(x, m, rmin=2.0)
    assert 0.0 <= y.min() and y.max() <= 1.0
    assert abs(y.sum() - x.sum()) < 0.02 * x.sum()


def test_projection_range():
    rng = np.random.default_rng(1)
    x = rng.random(100)
    y = heaviside_projection(x, beta=8.0, eta=0.5)
    assert 0.0 <= y.min() and y.max() <= 1.0
