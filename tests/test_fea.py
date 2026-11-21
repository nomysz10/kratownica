import numpy as np

from truss.mesh import make_rect_mesh
from truss.fea import assemble_stiffness, solve_linear


def test_cantilever_tip_deflection():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    K = assemble_stiffness(m, 2800.0, 0.35, 1.0)
    # utwierdzenie: wszystkie węzły na x=0 (lewa krawędź)
    left_nodes = np.where(m.nodes[:, 0] == 0.0)[0]
    fixed = np.array([d for n in left_nodes for d in (2 * n, 2 * n + 1)])
    f = np.zeros(2 * m.nodes.shape[0])
    # row-major: tip (col=nx, row=2) = 2*(nx+1)+nx -> y=5, x=200
    tip_node = 2 * (m.nx + 1) + m.nx
    f[2 * tip_node + 1] = -10.0
    u = solve_linear(K, f, fixed)
    assert abs(u[2 * tip_node + 1] + 114.2857) < 6.0  # EB: PL^3/(3EI)
