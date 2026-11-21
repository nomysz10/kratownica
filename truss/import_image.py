"""Import a bridge profile from a PNG and rasterize it to a binary mask."""

import numpy as np
from PIL import Image


def image_to_binary(path, span_mm, el_size, thr=128):
    """Rasterize a PNG profile to a boolean (ny, nx) mask.

    Bright pixels (> thr) are material. The image spans span_mm horizontally;
    vertical scale is the same mm/px. Returns (binary, height_mm).
    """
    a = np.array(Image.open(path).convert("L"))
    H, W = a.shape
    mm_per_px = span_mm / W
    height_mm = H * mm_per_px

    nx = int(round(span_mm / el_size))
    ny = int(round(height_mm / el_size))
    binary = np.zeros((ny, nx), dtype=bool)

    for j in range(ny):      # y index, 0 = bottom
        for i in range(nx):  # x index
            xc = (i + 0.5) * el_size
            yc = (j + 0.5) * el_size
            px = int(xc / mm_per_px)
            py = int((height_mm - yc) / mm_per_px)  # flip: image row 0 = top
            if 0 <= px < W and 0 <= py < H:
                binary[j, i] = a[py, px] > thr
    return binary, height_mm
