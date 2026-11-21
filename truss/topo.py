"""Density filter and Heaviside projection for SIMP topology optimization."""

import numpy as np


def _element_centers(mesh):
    """Centers (Nel, 2) of each element in mm."""
    nel = mesh.elements.shape[0]
    centers = np.zeros((nel, 2))
    for e, el in enumerate(mesh.elements):
        centers[e] = mesh.nodes[el].mean(axis=0)
    return centers


def density_filter(x, mesh, rmin):
    """Cone density filter: y_i = sum(w_ij x_j) / sum(w_ij),
    w_ij = max(0, rmin - dist(i,j)) between element centers.
    """
    centers = _element_centers(mesh)
    nel = x.shape[0]
    # pairwise distances between element centers
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    W = np.maximum(0.0, rmin - d)
    y = (W @ x) / W.sum(axis=1)
    return y


def heaviside_projection(x, beta, eta):
    """Smoothed Heaviside projection (0..1)."""
    y = (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    )
    return np.clip(y, 0.0, 1.0)
