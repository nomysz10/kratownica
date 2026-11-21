import numpy as np

from truss.mesh import make_rect_mesh
from truss.topo import density_filter, topology_optimize


def test_filter_preserves_volume():
    rng = np.random.default_rng(0)
    m = make_rect_mesh(120.0, 40.0, 2.0)
    x = rng.random(m.elements.shape[0])
    y = density_filter(x, m, rmin=2.0)
    assert 0.0 <= y.min() and y.max() <= 1.0
    assert abs(y.sum() - x.sum()) < 0.02 * x.sum()


def test_topo_cantilever_reduces_compliance():
    m = make_rect_mesh(120.0, 40.0, 2.0)
    rho, hist = topology_optimize(m, 2800.0, 0.35, volfrac=0.5,
                                  p=3.0, rmin=1.5, max_iter=60, move=0.2)
    assert hist[-1] < hist[0] * 0.8          # podatność spada
    assert abs(rho.mean() - 0.5) < 0.01      # frakcja utrzymana
    assert np.all(np.isfinite(rho)) and 0.0 <= rho.min() and rho.max() <= 1.0


def test_keepout_forces_void():
    m = make_rect_mesh(120.0, 40.0, 2.0)
    # otwór 40x20 nad jezdnią (y od 10 do 30), obciążenie na jezdni (y=0)
    keepout = (40.0, 80.0, 10.0, 30.0)
    rho, _ = topology_optimize(m, 2800.0, 0.35, volfrac=0.5,
                               p=3.0, rmin=1.5, max_iter=30, move=0.2,
                               keepout=keepout, y_load=0.0)
    centers = m.nodes[m.elements].mean(axis=1)
    mask = ((centers[:, 0] >= keepout[0]) & (centers[:, 0] <= keepout[1])
            & (centers[:, 1] >= keepout[2]) & (centers[:, 1] <= keepout[3]))
    assert rho[mask].max() < 1e-6            # dziura jest pusta
    assert abs(rho.mean() - 0.5) < 0.03      # masa wciąż ~volfrac
