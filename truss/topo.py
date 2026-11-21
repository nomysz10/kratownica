"""Density filter, Heaviside projection, and SIMP topology optimization."""

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from truss.fea import solve_linear, unit_element_stiffness


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


def heaviside_projection(x, beta, eta):
    """Smoothed Heaviside projection (0..1)."""
    y = (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    )
    return np.clip(y, 0.0, 1.0)


def _node_index(mesh, x, y):
    d = np.abs(mesh.nodes[:, 0] - x) + np.abs(mesh.nodes[:, 1] - y)
    return int(np.argmin(d))


def topology_optimize(mesh, E, nu, volfrac, *, p, rmin, max_iter, move):
    """SIMP + OC. Returns (rho, compliance_history).

    Simply-supported: left bottom corner pinned, right bottom corner roller,
    unit downward load at center of the top edge.
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

    # boundary conditions and load
    lb = _node_index(mesh, 0.0, 0.0)
    rb = _node_index(mesh, mesh.lx, 0.0)
    ct = _node_index(mesh, mesh.lx / 2.0, mesh.ly)
    fixed = np.array([2 * lb, 2 * lb + 1, 2 * rb + 1])
    f = np.zeros(n_dof)
    f[2 * ct + 1] = -1.0

    x = np.full(nel, volfrac)
    E_min = 1e-9 * E
    history = []

    for _ in range(max_iter):
        x_phys = np.asarray(H @ x).ravel() / Hs
        E_eff = E_min + x_phys ** p * (E - E_min)

        data = (E_eff[:, None, None] * Ke0[None, :, :]).ravel()
        K = sp.coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

        u = solve_linear(K, f, fixed)
        history.append(float(f @ u))

        ue = u[dofs]  # (Nel, 8)
        ce = np.einsum("ei,ij,ej->e", ue, Ke0, ue)
        dc = -p * x_phys ** (p - 1) * (E - E_min) * ce
        dc = np.asarray(H @ dc).ravel() / Hs  # sensitivity filter

        x = _oc_update(x, dc, volfrac, move, nel)

    x_phys = np.asarray(H @ x).ravel() / Hs
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
