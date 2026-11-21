"""Density filter and SIMP topology optimization."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from truss.constants import LOAD_PATCH
from truss.fea import (
    center_patch_load,
    recover_stress_variable,
    simply_supported_dofs,
    solve_linear,
    unit_element_stiffness,
)


def _element_centers(mesh):
    """Centers (Nel, 2) of each element in mm."""
    return mesh.nodes[mesh.elements].mean(axis=1)


def _filter_matrix(mesh, rmin):
    """Sparse cone-filter weight matrix H: H[i,j] = max(0, rmin - dist(i,j)).

    Includes self-weight rmin on the diagonal. Symmetric CSR.
    """
    centers = _element_centers(mesh)
    nel = centers.shape[0]
    tree = cKDTree(centers)
    pairs = tree.query_pairs(rmin, output_type="ndarray")
    if pairs.shape[0] > 0:
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(centers[i] - centers[j], axis=1)
        w = rmin - d
        rows = np.concatenate([i, j])
        cols = np.concatenate([j, i])
        vals = np.concatenate([w, w])
        H = sp.coo_matrix((vals, (rows, cols)), shape=(nel, nel)).tocsr()
    else:
        H = sp.csr_matrix((nel, nel))
    H = H + sp.identity(nel, format="csr") * rmin
    return H


def density_filter(x, mesh, rmin):
    """Cone density filter: y = (H x) / H.sum(axis=1)."""
    H = _filter_matrix(mesh, rmin)
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return np.asarray(H @ x).ravel() / Hs


def _node_index(mesh, x, y):
    d = np.abs(mesh.nodes[:, 0] - x) + np.abs(mesh.nodes[:, 1] - y)
    return int(np.argmin(d))


def _axial_compression(mesh, u, nu, x_phys, p, E_min, E):
    """Minimum principal stress per element (negative where compressed).

    Returns (Nel,) with negative values for compressive axial stress, so a
    compression penalty can thicken the compressed chord.
    """
    E_eff = E_min + x_phys ** p * (E - E_min)
    stress = recover_stress_variable(mesh, u, nu, 1.0, E_eff)  # (Nel,3) sxx,syy,sxy
    sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
    center = 0.5 * (sxx + syy)
    radius = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    s1 = center + radius
    s2 = center - radius
    return np.minimum(s1, s2)  # most compressive (negative)


def topology_optimize(mesh, E, nu, volfrac, *, p, rmin, max_iter, move,
                      keepout=None, y_load=None, on_iter=None,
                      compression_penalty=0.0):
    """SIMP + OC. Returns (rho, compliance_history).

    Simply-supported with a unit downward patch load at center. y_load=None
    means the top edge; y_load=0.0 loads the deck (through-truss). keepout =
    (x0, x1, y0, y1) in mm: elements whose center lies inside are forced to
    void. on_iter(x_phys, it) is called after each iteration. compression_penalty
    > 0 thickens compression members (adds a term to the sensitivity that keeps
    more material where axial compressive stress is high), which helps buckling.
    """
    nel = mesh.elements.shape[0]
    n_dof = 2 * mesh.nodes.shape[0]

    Ke0 = unit_element_stiffness(mesh, nu)

    # global DOF triplet structure (Nel, 8) and COO indices (Nel,8,8)
    dofs = np.stack([2 * mesh.elements, 2 * mesh.elements + 1], axis=-1).reshape(nel, 8)
    rows = np.repeat(dofs[:, :, None], 8, axis=2).ravel()
    cols = np.repeat(dofs[:, None, :], 8, axis=1).ravel()

    H = _filter_matrix(mesh, rmin)
    Hs = np.asarray(H.sum(axis=1)).ravel()

    # keepout mask
    if keepout is not None:
        centers = mesh.nodes[mesh.elements].mean(axis=1)
        x0, x1, y0, y1 = keepout
        kmask = ((centers[:, 0] >= x0) & (centers[:, 0] <= x1)
                 & (centers[:, 1] >= y0) & (centers[:, 1] <= y1))
    else:
        kmask = None

    # boundary conditions and load (patch load at center; y_load=None -> top edge)
    fixed = simply_supported_dofs(mesh)
    f = center_patch_load(mesh, LOAD_PATCH, 1.0, y_load=y_load)

    x = np.full(nel, volfrac)
    if kmask is not None:
        x[kmask] = 0.0
    E_min = 1e-9 * E
    history = []

    for _ in range(max_iter):
        x_phys = np.asarray(H @ x).ravel() / Hs
        if kmask is not None:
            x_phys[kmask] = 0.0
        E_eff = E_min + x_phys ** p * (E - E_min)

        data = (E_eff[:, None, None] * Ke0[None, :, :]).ravel()
        K = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

        u = solve_linear(K, f, fixed)
        history.append(float(f @ u))

        ue = u[dofs]  # (Nel, 8)
        ce = np.einsum("ei,ij,ej->e", ue, Ke0, ue)
        dc = -p * x_phys ** (p - 1) * (E - E_min) * ce

        # compression penalty: keep material where axial compressive stress is
        # high (thickens the compression chord against buckling). sig_c is
        # negative where compressed; add to make dc more negative there. Guard
        # against a singular matrix from over-penalizing a sparse field.
        if compression_penalty > 0.0:
            sig_c = _axial_compression(mesh, u, nu, x_phys, p, E_min, E)
            dc = dc + compression_penalty * sig_c * ce * x_phys

        dc = np.asarray(H @ dc).ravel() / Hs  # sensitivity filter

        x = _oc_update(x, dc, volfrac, move, nel)
        if kmask is not None:
            x[kmask] = 0.0
        if on_iter is not None:
            on_iter(x_phys, len(history) - 1)

    x_phys = np.asarray(H @ x).ravel() / Hs
    if kmask is not None:
        x_phys[kmask] = 0.0
    return x_phys, np.array(history)


def _oc_update(x, dc, volfrac, move, nel):
    """Optimality-criteria update (88-line style), move limit `move`."""
    target = volfrac * nel
    l1, l2 = 0.0, 1e9
    xnew = np.zeros_like(x)
    for _ in range(200):
        lmid = 0.5 * (l1 + l2)
        xnew = x * np.sqrt(-dc / lmid)
        xnew = np.clip(xnew, np.maximum(0.0, x - move), np.minimum(1.0, x + move))
        if xnew.sum() > target:
            l1 = lmid
        else:
            l2 = lmid
    return xnew
