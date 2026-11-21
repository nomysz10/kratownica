"""Plane-stress Q4 finite element assembly and solve.

DOF indexing: node n -> dofs (2n, 2n+1) = (ux, uy).
Units: mm, N, MPa.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# 2x2 Gauss quadrature points and weights (local coords xi, eta in [-1, 1])
_GP = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
_GW = np.array([1.0, 1.0])


def _shape_functions(xi, eta):
    """Bilinear Q4 shape functions and their derivatives.

    Returns N (4,), dN (4, 2) where dN[:, 0] = dN/dxi, dN[:, 1] = dN/deta.
    """
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])
    dN = 0.25 * np.array([
        [-(1 - eta), -(1 - xi)],
        [(1 - eta), -(1 + xi)],
        [(1 + eta), (1 + xi)],
        [-(1 + eta), (1 - xi)],
    ])
    return N, dN


def _plane_stress_D(E, nu):
    """Constitutive matrix for plane stress."""
    k = E / (1 - nu * nu)
    return k * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1 - nu) / 2.0],
    ])


def _B_matrix(dN, Jinv):
    """Strain-displacement matrix (3, 8) from dN (4,2) and Jinv (2,2)."""
    B = np.zeros((3, 8))
    for a in range(4):
        dNdx = Jinv @ dN[a]
        B[0, 2 * a] = dNdx[0]
        B[1, 2 * a + 1] = dNdx[1]
        B[2, 2 * a] = dNdx[1]
        B[2, 2 * a + 1] = dNdx[0]
    return B


def assemble_stiffness(mesh, E, nu, thickness):
    """Assemble global stiffness matrix (CSR) for plane-stress Q4."""
    n_dof = 2 * mesh.nodes.shape[0]
    K = sp.lil_matrix((n_dof, n_dof))
    D = _plane_stress_D(E, nu)

    for el in mesh.elements:
        coords = mesh.nodes[el]  # (4, 2)
        Ke = np.zeros((8, 8))
        for xi, wx in zip(_GP, _GW):
            for eta, wy in zip(_GP, _GW):
                _, dN = _shape_functions(xi, eta)
                J = dN.T @ coords  # (2, 2)
                detJ = np.linalg.det(J)
                Jinv = np.linalg.inv(J)
                B = _B_matrix(dN, Jinv)
                Ke += (B.T @ D @ B) * (detJ * wx * wy * thickness)

        dofs = np.array([[2 * n, 2 * n + 1] for n in el]).ravel()
        for i in range(8):
            for j in range(8):
                K[dofs[i], dofs[j]] += Ke[i, j]

    return K.tocsr()


def solve_linear(K, f, fixed_dofs):
    """Solve K u = f with prescribed zero displacement on fixed_dofs."""
    n = K.shape[0]
    free = np.setdiff1d(np.arange(n), fixed_dofs)
    u = np.zeros(n)
    u[free] = spla.spsolve(K[np.ix_(free, free)], f[free])
    return u


def compliance(u, f):
    """Compliance = f^T u."""
    return float(f @ u)


def recover_von_mises(mesh, u, E, nu, thickness):
    """Per-element von Mises stress at element centers (Nel,)."""
    D = _plane_stress_D(E, nu)
    n_el = mesh.elements.shape[0]
    vm = np.zeros(n_el)

    for e, el in enumerate(mesh.elements):
        coords = mesh.nodes[el]
        ue = u[np.array([[2 * n, 2 * n + 1] for n in el]).ravel()]
        # stress at element center (xi=eta=0)
        _, dN = _shape_functions(0.0, 0.0)
        J = dN.T @ coords
        Jinv = np.linalg.inv(J)
        B = _B_matrix(dN, Jinv)
        stress = D @ B @ ue  # (sxx, syy, sxy)
        sxx, syy, sxy = stress
        vm[e] = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)

    return vm
