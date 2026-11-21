import numpy as np

from truss.mesh import make_rect_mesh
from truss.visualize import deform_nodes, member_utilization


def test_deform_nodes():
    m = make_rect_mesh(10.0, 10.0, 5.0)  # 3x3 nodes
    u = np.zeros(2 * m.nodes.shape[0])
    u[1::2] = 1.0  # wszystkie przemieszczenia y = 1
    nodes = deform_nodes(m, u, scale=10.0)
    assert np.allclose(nodes[:, 1], m.nodes[:, 1] + 10.0)
    assert np.allclose(nodes[:, 0], m.nodes[:, 0])


def test_member_utilization():
    comp = {"is_compression": True, "n_unit": -10.0, "p_cr": 2000.0}
    tension = {"is_compression": False, "n_unit": 5.0, "p_cr": 100.0}
    # p_member = p_cr/(sf*|n_unit|) = 2000/(1.5*10)=133.3; util = 133.3/133.3=1.0
    assert abs(member_utilization(comp, p_max=133.333, sf=1.5) - 1.0) < 1e-3
    assert member_utilization(tension, p_max=100.0, sf=1.5) == 0.0
