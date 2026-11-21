import numpy as np

from truss.mesh import make_rect_mesh
from truss.verify import (
    compression_buckling_load,
    euler_critical_load,
    evaluate_shape,
    member_axial_force,
    member_buckling_load,
)


def test_euler_weak_axis():
    # w=10, H=4, L=100, E=2800: I = w*H^3/12 = 53.333 -> Pcr = 147.39 N
    pcr = euler_critical_load(width=10.0, height=4.0, length=100.0, E=2800.0)
    assert abs(pcr - 147.39) < 0.5


def test_euler_in_plane_weak_axis():
    # w=4, H=8, L=100, E=2800: H>w -> I = H*w^3/12 = 42.667 -> Pcr = 117.92 N
    pcr = euler_critical_load(width=4.0, height=8.0, length=100.0, E=2800.0)
    assert abs(pcr - 117.92) < 0.5


def test_compression_buckling_uses_brace_spacing():
    # wyboczenie boczne pasa ściskanego liczone na rozstawie stężeń (np. 100 mm),
    # a nie na krótkim odcinku w płaszczyźnie -> H wchodzi z H^3
    w, H, E, brace = 10.0, 4.0, 2800.0, 100.0
    pcr = compression_buckling_load(width=w, height=H, E=E, brace_spacing=brace)
    assert abs(pcr - 147.39) < 0.5
    # grubsze H dramatycznie poprawia: H=6 -> I = w*6^3/12 = 180 -> ~497 N
    pcr6 = compression_buckling_load(width=w, height=6.0, E=E, brace_spacing=brace)
    assert abs(pcr6 - 497.7) < 2.0


def test_member_buckling_checks_both_axes():
    # w=5, H=7, L=246, brace=50, E=2800
    # w płaszczyźnie: I = H*w^3/12 = 7*125/12 = 72.9 -> Pcr_in = pi^2*E*I/246^2 ~ 33 N
    # poza płaszczyzną: I = w*H^3/12 = 5*343/12 = 142.9 -> Pcr_out ~ 1580 N
    p_in, p_out = member_buckling_load(width=5.0, height=7.0, member_length=246.0,
                                       brace_spacing=50.0, E=2800.0)
    assert abs(p_in - 33.3) < 2.0
    assert abs(p_out - 1580.0) < 60.0
    # słabszy jest tryb w płaszczyźnie (cienki, długi pas)
    assert p_in < p_out


def test_member_axial_force():
    n = member_axial_force(axis=np.array([1.0, 0.0]),
                           stress=np.array([10.0, 0.0, 0.0]),  # sxx=10 MPa
                           width=10.0, height=4.0)
    assert abs(n - 400.0) < 1e-9


def test_solid_beam_load_limit():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    binary = np.ones((m.ny, m.nx), dtype=bool)
    # load_patch=2.5 -> single center node, so EB point-load formula holds
    r = evaluate_shape(m, binary, 2.5, 2800.0, 0.35, 1.0, 20.0, 1.5, 20.0, load_patch=2.5)
    # EB: δ(1N) = L^3/(48EI) = 0.7143 mm -> p_ugiecie = 20/0.7143 = 28.0 N
    assert abs(r["p_ugiecie"] - 28.0) < 1.7
    assert abs(r["mass_g"] - 200 * 10 * 1 * 1.24e-3) < 0.01
    assert r["p_max"] == min(r["p_ugiecie"], r["p_naprezenie"], r["p_wyboczenie"])
