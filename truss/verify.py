"""Verification of the final (thresholded) shape: deflection, stress, buckling."""

import numpy as np

from truss.constants import LOAD_PATCH, RHO_PLA, SUPPORT_PATCH
from truss.fea import (
    assemble_stiffness_variable,
    center_patch_load,
    recover_stress_variable,
    simply_supported_dofs,
    solve_linear,
)
from truss.postprocess import skeletonize_members


def simulate_binary_shape(mesh, binary, E, nu, H, load, load_patch=LOAD_PATCH, y_load=None):
    """Run plane-stress FE on a binarized element field.

    Void elements get E=1e-9*E. Load spread over nodes at y=y_load (top edge
    by default) within +-load_patch/2 of center; simply-supported at the two
    bottom corners. Returns (u, von_mises).
    """
    b = np.asarray(binary, dtype=bool).ravel()
    E_eff = np.where(b, E, 1e-9 * E)

    K = assemble_stiffness_variable(mesh, nu, H, E_eff)

    fixed = simply_supported_dofs(mesh)
    f = center_patch_load(mesh, load_patch, load, y_load=y_load)

    u = solve_linear(K, f, fixed)

    stress = recover_stress_variable(mesh, u, nu, H, E_eff)
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
    return u, vm


def center_deflection(u, mesh, center_x, y_load=None):
    """Vertical deflection (mm) at the node at y=y_load (top edge by default)
    nearest center_x."""
    if y_load is None:
        y_load = mesh.ly
    row = mesh.nodes[:, 1] == y_load
    row_nodes = np.where(row)[0]
    x = mesh.nodes[row_nodes, 0]
    idx = row_nodes[np.argmin(np.abs(x - center_x))]
    return abs(u[2 * idx + 1])


def _stress_peak_excluding_supports(mesh, vm, load_patch=LOAD_PATCH):
    """Peak von Mises with elements under the load and support patches excluded
    (avoids point-load / point-support stress singularities)."""
    centers = mesh.nodes[mesh.elements].mean(axis=1)
    cx = mesh.lx / 2.0
    near_load = (np.abs(centers[:, 0] - cx) <= load_patch / 2.0) & (
        centers[:, 1] >= mesh.ly - load_patch / 2.0
    )
    near_support = (
        (centers[:, 0] <= SUPPORT_PATCH / 2.0) | (centers[:, 0] >= mesh.lx - SUPPORT_PATCH / 2.0)
    ) & (centers[:, 1] <= SUPPORT_PATCH / 2.0)
    mask = ~(near_load | near_support)
    return vm[mask].max() if mask.any() else vm.max()


def euler_critical_load(width, height, length, E, K=1.0):
    """Euler buckling load (N) about the weak axis.

    Cross-section width (in-plane) x height (out-of-plane, H). The weak axis
    is min(width, height): I = min^3 * max / 12, so it covers both H < w
    (out-of-plane buckling) and H > w (in-plane buckling).
    """
    h_min, h_max = sorted((width, height))
    I = h_min ** 3 * h_max / 12.0
    return np.pi ** 2 * E * I / (K * length) ** 2


def member_axial_force(axis, stress, width, height):
    """Axial force (N) = (axis·σ·axis) * area, stress = (sxx, syy, sxy)."""
    ax, ay = axis
    sxx, syy, sxy = stress
    sig_axial = sxx * ax * ax + syy * ay * ay + 2 * sxy * ax * ay
    return sig_axial * width * height


def evaluate_shape(mesh, binary, pixel_size, E, nu, H, sigma_allow, sf, max_deflection,
                   load_patch=LOAD_PATCH, y_load=None):
    """Load limits (N) and mass (g) for a thresholded shape.

    Returns dict with mass_g, deflection_at_1N, p_ugiecie, p_naprezenie,
    p_wyboczenie, p_max, breakdown, and visualization fields (u, vm, members).
    """
    b = np.asarray(binary, dtype=bool)
    u, vm = simulate_binary_shape(mesh, b, E, nu, H, load=1.0, load_patch=load_patch,
                                  y_load=y_load)

    delta = center_deflection(u, mesh, mesh.lx / 2.0)
    p_ugiecie = max_deflection / delta
    p_naprezenie = sigma_allow / _stress_peak_excluding_supports(mesh, vm, load_patch)

    # mass
    area_px = pixel_size ** 2
    mass_g = b.sum() * area_px * H * RHO_PLA

    # buckling: compression members from skeleton, peak axial force at unit load
    members = skeletonize_members(b, pixel_size)
    stress = recover_stress_variable(mesh, u, nu, H, np.where(b.ravel(), E, 1e-9 * E))
    centers = mesh.nodes[mesh.elements].mean(axis=1)

    p_wyboczenie = np.inf
    member_data = []
    for m in members:
        if m.length <= 0 or m.width <= 0:
            continue
        axis = m.points[-1] - m.points[0]
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        # peak compressive axial force sampled along the member (not one point)
        n_unit = np.inf
        for pt in m.points[:: max(1, len(m.points) // 8)]:
            e = int(np.argmin(np.linalg.norm(centers - pt, axis=1)))
            n_unit = min(n_unit, member_axial_force(axis, stress[e], m.width, H))
        p_cr = euler_critical_load(m.width, H, m.length, E)
        member_data.append({
            "points": m.points,
            "width": m.width,
            "length": m.length,
            "n_unit": n_unit,
            "p_cr": p_cr,
            "is_compression": n_unit < 0.0,
        })
        if n_unit < 0.0:  # compression
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
        "u": u,
        "vm": vm,
        "members": member_data,
    }
