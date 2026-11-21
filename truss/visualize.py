"""Visualization: deformed-mesh stress and buckling-utilization plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
import numpy as np
from scipy.spatial import cKDTree


def deform_nodes(mesh, u, scale):
    """Node positions displaced by scale * u (u has 2 DOF per node)."""
    return mesh.nodes + scale * u.reshape(-1, 2)


def member_utilization(member, p_max, sf):
    """Buckling utilization of a member at load p_max (0 for tension).

    utilization = p_max * sf * |n_unit| / p_cr; >= 1.0 means buckling.
    """
    if not member["is_compression"]:
        return 0.0
    p_cr = member["p_cr"]
    n_unit = abs(member["n_unit"])
    if p_cr <= 0 or n_unit <= 0:
        return 0.0
    return p_max * sf * n_unit / p_cr


def _deformed_solid_quads(mesh, binary, u, scale):
    nodes = deform_nodes(mesh, u, scale)
    verts, vals = [], []
    solid = np.asarray(binary).ravel()
    for e, el in enumerate(mesh.elements):
        if not solid[e]:
            continue
        q = nodes[el]  # (4,2) in CCW order
        verts.append(q)
        vals.append(e)
    return verts, vals


def plot_stress(path, mesh, binary, u, vm, scale, title):
    """Deformed geometry colored by von Mises stress (MPa)."""
    verts, elem_idx = _deformed_solid_quads(mesh, binary, u, scale)
    if not verts:
        return
    vals = np.array([vm[i] for i in elem_idx])
    fig, ax = plt.subplots(figsize=(14, 4))
    pc = PolyCollection(verts, array=vals, cmap="turbo", edgecolors="none")
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="von Mises [MPa]")
    ax.set_title(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_buckling(path, mesh, members, u, scale, p_max, sf, title):
    """Members colored by buckling utilization (0..1)."""
    tree = cKDTree(mesh.nodes)
    disp = u.reshape(-1, 2)
    segs, cols = [], []
    for m in members:
        pts = m["points"]
        _, idx = tree.query(pts)
        pts_d = pts + scale * disp[idx]
        util = member_utilization(m, p_max, sf)
        for a, b in zip(pts_d[:-1], pts_d[1:]):
            segs.append([a, b])
            cols.append(util)
    if not segs:
        return
    fig, ax = plt.subplots(figsize=(14, 4))
    lc = LineCollection(segs, cmap="coolwarm", norm=Normalize(0.0, 1.0))
    lc.set_array(np.array(cols))
    lc.set_linewidth(3)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    fig.colorbar(lc, ax=ax, label="wykorzystanie wyboczeniowe")
    ax.set_title(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def animate_simp(frames, binary_shape, el_size, path, duration=4.0):
    """Assemble a GIF of the density field evolving (frames: list of (ny,nx))."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    if not frames:
        return
    ny, nx = binary_shape
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(np.zeros((ny, nx)), cmap="gray_r", origin="lower", vmin=0.0, vmax=1.0,
                   extent=[0, nx * el_size, 0, ny * el_size])
    ax.set_xlabel("span [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title("iteracja 0")
    fig.colorbar(im, ax=ax, label="gęstość")

    def update(i):
        im.set_array(frames[i])
        ax.set_title(f"iteracja {i}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=int(duration * 1000 / len(frames)))
    ani.save(path, writer=PillowWriter(fps=len(frames) / duration))
    plt.close(fig)
