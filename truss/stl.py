"""Binary STL writer and voxel extrusion of a 2D binary material field."""

import numpy as np


def _quad(tris, a, b, c, d, n_out):
    """Emit two triangles for a quad, wound so the normal points n_out."""
    for v0, v1, v2 in ((a, b, c), (a, c, d)):
        nrm = np.cross(v1 - v0, v2 - v0)
        if np.dot(nrm, n_out) < 0.0:
            v1, v2 = v2, v1
        tris.append(np.array([v0, v1, v2]))


def extrude_binary_to_triangles(binary, el_size, H):
    """Extrude a boolean grid (ny, nx) into an outward-wound mesh (M,3,3).

    Each solid cell is an el_size x el_size x H box; faces shared with a
    neighboring solid cell are omitted, so the mesh is watertight, preserves
    holes, and its volume equals n_solid * el_size^2 * H.
    """
    b = np.asarray(binary, dtype=bool)
    ny, nx = b.shape
    tris = []

    for i in range(ny):
        for j in range(nx):
            if not b[i, j]:
                continue
            x0, x1 = j * el_size, (j + 1) * el_size
            y0, y1 = i * el_size, (i + 1) * el_size
            bl = np.array([x0, y0, 0.0])
            br = np.array([x1, y0, 0.0])
            tr = np.array([x1, y1, 0.0])
            tl = np.array([x0, y1, 0.0])
            blt = np.array([x0, y0, H])
            brt = np.array([x1, y0, H])
            trt = np.array([x1, y1, H])
            tlt = np.array([x0, y1, H])

            # top (+Z) and bottom (-Z) caps: always exterior (single layer)
            _quad(tris, blt, brt, trt, tlt, np.array([0.0, 0.0, 1.0]))
            _quad(tris, bl, br, tr, tl, np.array([0.0, 0.0, -1.0]))
            # side walls only at material/void or domain boundaries
            if j == 0 or not b[i, j - 1]:
                _quad(tris, bl, tl, tlt, blt, np.array([-1.0, 0.0, 0.0]))
            if j == nx - 1 or not b[i, j + 1]:
                _quad(tris, br, brt, trt, tr, np.array([1.0, 0.0, 0.0]))
            if i == 0 or not b[i - 1, j]:
                _quad(tris, bl, blt, brt, br, np.array([0.0, -1.0, 0.0]))
            if i == ny - 1 or not b[i + 1, j]:
                _quad(tris, tl, tr, trt, tlt, np.array([0.0, 1.0, 0.0]))

    return np.array(tris) if tris else np.zeros((0, 3, 3))


def mesh_volume(triangles):
    """Signed volume (mm^3) via divergence theorem (outward-wound triangles)."""
    tris = np.asarray(triangles, dtype=float)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)


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
