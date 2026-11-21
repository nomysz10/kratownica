"""Outer optimization loop over truss depth D and height H."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion

from truss import constants as C
from truss.mesh import make_rect_mesh
from truss.postprocess import (
    enforce_min_width,
    keep_largest_component,
    skeletonize_members,
    threshold_binary,
)
from truss.topo import topology_optimize
from truss.verify import evaluate_shape


@dataclass
class Result:
    D: float
    H: float
    el_size: float
    rho: np.ndarray
    binary: np.ndarray
    p_max: float
    mass_g: float
    deflection_at_1N: float
    breakdown: dict
    members: list
    compliance_history: np.ndarray


def run_optimization(span, max_mass_g, max_deflection, d_range, h_range, el_size, max_iter=60):
    """Sweep (D, H), run SIMP + post-process + verify, return best Result."""
    mass_mm3 = max_mass_g / C.RHO_PLA
    rmin = max(C.FILTER_RMIN, 1.5 * el_size)

    Ds = np.arange(d_range[0], d_range[1] + 1e-9, el_size)
    Hs = np.arange(h_range[0], h_range[1] + 1e-9, 1.0)

    best = None
    for D in Ds:
        ny = int(round(D / el_size))
        if abs(D - ny * el_size) > 1e-6:
            continue  # D must be a multiple of el_size for a square mesh
        for H in Hs:
            volfrac = min(1.0, mass_mm3 / (span * D * H))
            mesh = make_rect_mesh(span, D, el_size)
            rho, hist = topology_optimize(
                mesh, C.E, C.NU, volfrac,
                p=C.SIMP_P, rmin=rmin, max_iter=max_iter, move=0.2,
            )
            binary = threshold_binary(rho, 0.5).reshape(mesh.ny, mesh.nx)
            binary = keep_largest_component(binary)
            binary = enforce_min_width(binary, el_size, C.MIN_WIDTH)

            # enforce hard mass constraint (erode if the dilation overshot)
            mass_g = binary.sum() * el_size ** 2 * H * C.RHO_PLA
            while mass_g > max_mass_g:
                binary = binary_erosion(binary)
                if not binary.any():
                    break
                mass_g = binary.sum() * el_size ** 2 * H * C.RHO_PLA
            if not binary.any():
                continue

            r = evaluate_shape(
                mesh, binary, el_size, C.E, C.NU, H,
                C.SIGMA_ALLOW, C.SAFETY_FACTOR, max_deflection,
            )
            res = Result(
                D=D, H=H, el_size=el_size, rho=rho, binary=binary,
                p_max=r["p_max"], mass_g=r["mass_g"],
                deflection_at_1N=r["deflection_at_1N"],
                breakdown=r["breakdown"],
                members=skeletonize_members(binary, el_size),
                compliance_history=hist,
            )
            if best is None or res.p_max > best.p_max:
                best = res

    return best
