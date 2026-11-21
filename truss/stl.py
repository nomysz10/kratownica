"""Binary STL writer and contour extrusion."""

import numpy as np


def _triangulate_fan(poly_xy):
    """Triangulate a planar polygon (N,2) with a fan from vertex 0."""
    n = len(poly_xy)
    tris = []
    for i in range(1, n - 1):
        tris.append(poly_xy[[0, i, i + 1]])
    return np.array(tris) if tris else np.zeros((0, 3, 2))


def extrude_contour_to_triangles(polygon, height):
    """Extrude a planar polygon (N,2) by `height` in Z into triangles (M,3,3).

    Bottom cap (z=0) + top cap (z=height) + side walls. Normals are computed
    per triangle in write_binary_stl, so orientation here only fixes winding.
    """
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)
    bottom_xy = _triangulate_fan(poly)
    top_xy = _triangulate_fan(poly)  # same xy, z=height

    tris = []
    # bottom cap, winding so normal points -Z
    for t in bottom_xy:
        tris.append(np.column_stack([t, np.zeros(3)]))
    # top cap, reversed winding so normal points +Z
    for t in top_xy:
        tris.append(np.column_stack([t[::-1], np.full(3, height)]))
    # side walls: each edge = 2 triangles
    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i + 1) % n]
        b0 = np.array([p0[0], p0[1], 0.0])
        b1 = np.array([p1[0], p1[1], 0.0])
        t0 = np.array([p0[0], p0[1], height])
        t1 = np.array([p1[0], p1[1], height])
        tris.append(np.array([b0, b1, t0]))
        tris.append(np.array([t0, b1, t1]))

    return np.array(tris)


def write_binary_stl(path, triangles):
    """Write triangles (M,3,3) to a binary STL file."""
    tris = np.asarray(triangles, dtype=np.float64)
    n = tris.shape[0]

    with open(path, "wb") as f:
        f.write(b" " * 80)
        f.write(np.uint32(n).tobytes())
        for tri in tris:
            v0, v1, v2 = tri
            normal = np.cross(v1 - v0, v2 - v0)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            f.write(normal.astype(np.float32).tobytes())
            for v in (v0, v1, v2):
                f.write(v.astype(np.float32).tobytes())
            f.write(np.uint16(0).tobytes())
