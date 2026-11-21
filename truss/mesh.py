"""Uniform rectangular Q4 mesh with square elements."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    nodes: np.ndarray      # (N, 2) mm
    elements: np.ndarray   # (Nel, 4) int, Q4 connectivity
    nx: int                # elements along x
    ny: int                # elements along y
    lx: float              # domain width mm
    ly: float              # domain height mm
    el_size: float         # element edge mm


def make_rect_mesh(lx: float, ly: float, el_size: float) -> Mesh:
    """Build a row-major rectangular mesh with square Q4 elements.

    Row-major: node index = row * (nx + 1) + col, x varies fastest.
    Raises ValueError if lx or ly is not an integer multiple of el_size.
    """
    if lx % el_size != 0.0 or ly % el_size != 0.0:
        raise ValueError("lx and ly must be integer multiples of el_size")

    nx = int(round(lx / el_size))
    ny = int(round(ly / el_size))

    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys)  # row-major: row = y, col = x
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1) + 1
            n3 = n0 + (nx + 1)
            elements.append([n0, n1, n2, n3])  # counter-clockwise
    elements = np.array(elements, dtype=int)

    return Mesh(
        nodes=nodes,
        elements=elements,
        nx=nx,
        ny=ny,
        lx=lx,
        ly=ly,
        el_size=el_size,
    )
