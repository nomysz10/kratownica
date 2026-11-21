"""Verification of the final (thresholded) shape: deflection, stress, buckling."""

import numpy as np

from truss.constants import LOAD_PATCH, RHO_PLA
from truss.fea import (
    assemble_stiffness_variable,
    recover_stress_variable,
    solve_linear,
)
from truss.postprocess import skeletonize_members


def _node_index(mesh, x, y):
    d = np.abs(mesh.nodes[:, 0] - x) + np.abs(mesh.nodes[:, 1] - y)
    return int(np.argmin(d))


def simulate_binary_shape(mesh, binary, E, nu, H, load):
    """Run plane-stress FE on a binarized element field.

    Void elements get E=1e-9*E. Unit load spread over top-edge nodes within
    +-LOAD_PATCH/2 of center; left bottom corner pinned, right bottom roller.
    Returns (u, von_mises).
    """
    b = np.asarray(binary, dtype=bool).ravel()
    E_eff = np.where(b, E, 1e-9 * E)

    K = assemble_stiffness_variable(mesh, nu, H, E_eff)

    n_dof = 2 * mesh.nodes.shape[0]
    lb = _node_index(mesh, 0.0, 0.0)
    rb = _node_index(mesh, mesh.lx, 0.0)
    fixed = np.array([2 * lb, 2 * lb + 1, 2 * rb + 1])

    top = mesh.nodes[:, 1] == mesh.ly
    cx = mesh.lx / 2.0
    half = LOAD_PATCH / 2.0
    load_nodes = np.where(top & (np.abs(mesh.nodes[:, 0] - cx) <= half))[0]

    f = np.zeros(n_dof)
    f[2 * load_nodes + 1] = -load / load_nodes.shape[0]

    u = solve_linear(K, f, fixed)

    stress = recover_stress_variable(mesh, u, nu, H, E_eff)
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
    return u, vm


def center_deflection(u, mesh, center_x):
    """Vertical deflection (mm) at the top-edge node nearest center_x."""
    top = mesh.nodes[:, 1] == mesh.ly
    top_nodes = np.where(top)[0]
    x = mesh.nodes[top_nodes, 0]
    idx = top_nodes[np.argmin(np.abs(x - center_x))]
    return abs(u[2 * idx + 1])


def euler_critical_load(width, height, length, E, K=1.0):
    """Euler buckling load (N) about the weak (out-of-plane) axis.

    Cross-section width (in-plane) x height (out-of-plane, H<w):
    I = width * height^3 / 12.
    """
    I = width * height ** 3 / 12.0
    return np.pi ** 2 * E * I / (K * length) ** 2


def member_axial_force(axis, stress, width, height):
    """Axial force (N) = (axis·σ·axis) * area, stress = (sxx, syy, sxy)."""
    ax, ay = axis
    sxx, syy, sxy = stress
    sig_axial = sxx * ax * ax + syy * ay * ay + 2 * sxy * ax * ay
    return sig_axial * width * height


def evaluate_shape(mesh, binary, pixel_size, E, nu, H, sigma_allow, sf, max_deflection):
    """Load limits (N) and mass (g) for a thresholded shape.

    Returns dict with mass_g, deflection_at_1N, p_ugiecie, p_naprezenie,
    p_wyboczenie, p_max, breakdown.
    """
    b = np.asarray(binary, dtype=bool)
    u, vm = simulate_binary_shape(mesh, b, E, nu, H, load=1.0)

    delta = center_deflection(u, mesh, mesh.lx / 2.0)
    p_ugiecie = max_deflection / delta
    p_naprezenie = sigma_allow / vm.max()  # linear: vm scales with load

    # mass
    area_px = pixel_size ** 2
    mass_g = b.sum() * area_px * H * RHO_PLA

    # buckling: compression members from skeleton, axial force at unit load
    members = skeletonize_members(b, pixel_size)
    stress = recover_stress_variable(mesh, u, nu, H, np.where(b.ravel(), E, 1e-9 * E))
    centers = mesh.nodes[mesh.elements].mean(axis=1)

    p_wyboczenie = np.inf
    for m in members:
        if m.length <= 0 or m.width <= 0:
            continue
        axis = m.points[-1] - m.points[0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        centroid = m.points.mean(axis=0)
        e = int(np.argmin(np.linalg.norm(centers - centroid, axis=1)))
        n_unit = member_axial_force(axis, stress[e], m.width, H)
        if n_unit < 0.0:  # compression
            p_cr = euler_critical_load(m.width, H, m.length, E)
            p_member = p_cr / (sf * abs(n_unit))
            p_wyboczenie = min(p_wyboczenie, p_member)

    p_max = min(p_ugiecie, p_naprezenie, p_wyboczenie)
    limits = {"ugiecie": p_ugiecie, "naprezenie": p_naprezenie, "wyboczenie": p_wyboczenie}
    governing = min(limits, key=limits.get)

    return {
        "mass_g": mass_g,
        "deflection_at_1N": delta,
        "p_ugiecie": p_ugiecie,
        "p_naprezenie": p_naprezenie,
        "p_wyboczenie": p_wyboczenie,
        "p_max": p_max,
        "breakdown": {**limits, "governs": governing},
    }
