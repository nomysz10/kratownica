"""Post-processing of SIMP density field into a printable shape."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import binary_dilation, skeletonize
from skimage.measure import find_contours, label, regionprops


def threshold_binary(rho, thr):
    """Element-centered boolean grid from a density field (same shape as rho)."""
    return np.asarray(rho) > thr


def keep_largest_component(binary):
    """Keep only the largest connected component (4-connectivity)."""
    lab, n = label(binary, connectivity=1, return_num=True)
    if n == 0:
        return binary.copy()
    sizes = np.array([(lab == i).sum() for i in range(1, n + 1)])
    keep = int(np.argmax(sizes)) + 1
    return lab == keep


def _edt(binary):
    return distance_transform_edt(binary)


def min_feature_width(binary, pixel_size):
    """Minimum member width (mm): 2 * max distance-transform along skeleton."""
    if not binary.any():
        return 0.0
    skel = skeletonize(binary)
    dt = _edt(binary)
    return 2.0 * dt[skel].max() * pixel_size


def enforce_min_width(binary, pixel_size, min_mm):
    """Dilate until the thinnest member reaches min_mm."""
    out = binary.copy()
    max_iters = int(np.ceil(min_mm / pixel_size)) + 2
    for _ in range(max_iters):
        if min_feature_width(out, pixel_size) >= min_mm:
            break
        out = binary_dilation(out)
    return out


def extract_contours(binary, pixel_size):
    """Closed polygons (N,2) mm for each contour at the 0/1 boundary."""
    polys = find_contours(binary.astype(float), 0.5)
    return [np.column_stack([p[:, 1], p[:, 0]]) * pixel_size for p in polys]


@dataclass
class Member:
    points: np.ndarray   # skeleton polyline (M, 2) mm
    width: float         # avg member width mm
    length: float        # polyline length mm


def skeletonize_members(binary, pixel_size):
    """Split the skeleton into segments at junctions; return Member list.

    Length is measured after extending each endpoint to the material
    boundary along the member axis (skeletonize otherwise trims the ends),
    width is the median thickness along the medial-axis (untrimmed) points.
    """
    if not binary.any():
        return []

    skel = skeletonize(binary)
    dt = _edt(binary)
    rows, cols = np.nonzero(skel)
    pts = set(zip(rows.tolist(), cols.tolist()))

    def neighbors(p):
        r, c = p
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                q = (r + dr, c + dc)
                if q in pts:
                    out.append(q)
        return out

    def extend(path, path_set, at_start):
        while True:
            a = path[0] if at_start else path[-1]
            b = path[1] if at_start else path[-2]
            d = np.array(a) - np.array(b)
            if abs(d[0]) >= abs(d[1]):
                step = np.array([int(np.sign(d[0])), 0])
            else:
                step = np.array([0, int(np.sign(d[1]))])
            cand = np.array(a) + step
            cr, cc = int(cand[0]), int(cand[1])
            if cr < 0 or cc < 0 or cr >= binary.shape[0] or cc >= binary.shape[1]:
                break
            if not binary[cr, cc] or (cr, cc) in path_set:
                break
            if at_start:
                path.insert(0, (cr, cc))
            else:
                path.append((cr, cc))
            path_set.add((cr, cc))

    deg = {p: len(neighbors(p)) for p in pts}
    starts = [p for p, d in deg.items() if d == 1]
    if not starts:
        starts = [p for p, d in deg.items() if d == 0]
    if not starts:
        starts = list(pts)

    members = []
    visited = set()
    for s in starts:
        if s in visited:
            continue
        path = [s]
        visited.add(s)
        cur = s
        while True:
            nxt = [q for q in neighbors(cur) if q not in visited]
            if not nxt:
                break
            n = nxt[0]
            path.append(n)
            visited.add(n)
            cur = n
            if deg[n] != 2:  # reached endpoint or junction
                break

        if len(path) < 2:
            continue

        medial = list(path)
        extend(path, visited, at_start=True)
        extend(path, visited, at_start=False)

        mm = np.array([[c, r] for r, c in path], dtype=float) * pixel_size
        length = float(np.sum(np.linalg.norm(np.diff(mm, axis=0), axis=1)))
        width = 2.0 * float(np.median([dt[r, c] for r, c in medial])) * pixel_size
        members.append(Member(points=mm, width=width, length=length))

    return members
