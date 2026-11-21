"""Outer optimization loop over truss depth D and height H."""

from dataclasses import dataclass
import os

import numpy as np
from scipy.ndimage import binary_erosion

from truss import constants as C
from truss.mesh import make_rect_mesh
from truss.postprocess import (
    enforce_min_width,
    keep_largest_component,
    smooth_binary,
    threshold_binary,
)
from truss.topo import topology_optimize
from truss.verify import evaluate_shape


@dataclass
class Result:
    D: float
    H: float
    el_size: float          # verification-mesh element size (el_size/factor)
    rho: np.ndarray
    binary: np.ndarray      # smoothed, filleted verification geometry
    mesh: object            # verification mesh (Mesh)
    p_max: float
    mass_g: float
    deflection_at_1N: float
    breakdown: dict
    members: list           # per-member dicts with n_unit, p_cr, is_compression
    u: np.ndarray           # displacement field at unit load
    vm: np.ndarray          # per-element von Mises at unit load
    compliance_history: np.ndarray


def _eval_case(case):
    """Evaluate one (D, H) combination; returns Result or None."""
    (span, max_mass_g, max_deflection, el_size, max_iter, smooth_factor,
     mass_mm3, rmin, D, H, keepout, y_load) = case
    volfrac = min(1.0, mass_mm3 / (span * D * H))
    mesh = make_rect_mesh(span, D, el_size)
    rho, hist = topology_optimize(
        mesh, C.E, C.NU, volfrac,
        p=C.SIMP_P, rmin=rmin, max_iter=max_iter, move=0.2,
        keepout=keepout, y_load=y_load,
    )
    binary = threshold_binary(rho, 0.5).reshape(mesh.ny, mesh.nx)
    binary = keep_largest_component(binary)
    binary = enforce_min_width(binary, el_size, C.MIN_WIDTH)
    binary = keep_largest_component(binary)

    # hard mass constraint: erode if the min-width dilation overshot
    mass_g = binary.sum() * el_size ** 2 * H * C.RHO_PLA
    while mass_g > max_mass_g:
        binary = binary_erosion(binary)
        binary = keep_largest_component(binary)
        if not binary.any():
            return None
        mass_g = binary.sum() * el_size ** 2 * H * C.RHO_PLA

    # fillet + refine: upscale+smooth so stress at joints is resolved
    fine = smooth_binary(binary, factor=smooth_factor)
    el_fine = el_size / smooth_factor
    mesh_fine = make_rect_mesh(span, D, el_fine)
    r = evaluate_shape(
        mesh_fine, fine, el_fine, C.E, C.NU, H,
        C.SIGMA_ALLOW, C.SAFETY_FACTOR, max_deflection,
        load_patch=C.LOAD_PATCH, y_load=y_load,
    )
    return Result(
        D=D, H=H, el_size=el_fine, rho=rho, binary=fine, mesh=mesh_fine,
        p_max=r["p_max"], mass_g=r["mass_g"],
        deflection_at_1N=r["deflection_at_1N"],
        breakdown=r["breakdown"], members=r["members"],
        u=r["u"], vm=r["vm"], compliance_history=hist,
    )


def run_optimization(span, max_mass_g, max_deflection, d_range, h_range, el_size,
                     max_iter=60, smooth_factor=4, n_workers=1, keepout=None,
                     y_load=None, on_case_done=None):
    """Sweep (D, H), run SIMP + post-process + verify, return best Result.

    n_workers > 1 parallelizes the (D,H) sweep with multiprocessing.
    on_case_done(result_or_none) is called after each case completes.
    """
    mass_mm3 = max_mass_g / C.RHO_PLA
    rmin = max(C.FILTER_RMIN, 1.5 * el_size)

    Ds = np.arange(d_range[0], d_range[1] + 1e-9, el_size)
    Hs = np.arange(h_range[0], h_range[1] + 1e-9, 1.0)
    cases = []
    for D in Ds:
        ny = int(round(D / el_size))
        if abs(D - ny * el_size) > 1e-6:
            continue
        for H in Hs:
            cases.append((span, max_mass_g, max_deflection, el_size, max_iter,
                          smooth_factor, mass_mm3, rmin, D, H, keepout, y_load))

    results = []
    if n_workers > 1:
        import multiprocessing as mp
        with mp.Pool(n_workers) as pool:
            for res in pool.imap_unordered(_eval_case, cases):
                results.append(res)
                if on_case_done is not None:
                    on_case_done(res)
    else:
        for c in cases:
            res = _eval_case(c)
            results.append(res)
            if on_case_done is not None:
                on_case_done(res)

    valid = [r for r in results if r is not None]
    if not valid:
        return None
    # competition score = load / mass (N per gram), not raw load
    return max(valid, key=lambda r: r.p_max / r.mass_g)
