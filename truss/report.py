"""Report writer: geometry PNG, extruded STL, text summary."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from truss.stl import extrude_binary_to_triangles, write_binary_stl


def write_report(result, outdir):
    """Write PNG/STL/txt for a Result. Returns dict of paths."""
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, "geometria.png")
    stl_path = os.path.join(outdir, "mostek.stl")
    txt_path = os.path.join(outdir, "raport.txt")

    # geometry PNG: binary as image (rows = depth, cols = span)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(result.binary.astype(float), cmap="gray_r", origin="lower",
              extent=[0, result.binary.shape[1] * result.el_size,
                      0, result.binary.shape[0] * result.el_size])
    ax.set_xlabel("span [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(f"mostek D={result.D:.1f} H={result.H:.1f}")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    # STL: voxel-extrude the material region (holes preserved)
    tris = extrude_binary_to_triangles(result.binary, result.el_size, result.H)
    write_binary_stl(stl_path, tris)

    # text summary
    b = result.breakdown
    lines = [
        f"D = {result.D:.2f} mm",
        f"H = {result.H:.2f} mm",
        f"el_size = {result.el_size:.2f} mm",
        f"mass_g = {result.mass_g:.2f}",
        f"deflection_at_1N = {result.deflection_at_1N:.4f} mm",
        f"p_max = {result.p_max:.2f} N",
        f"p_ugiecie = {b['ugiecie']:.2f} N",
        f"p_naprezenie = {b['naprezenie']:.2f} N",
        f"p_wyboczenie = {b['wyboczenie']:.2f} N",
        f"governs = {b['governs']}",
    ]
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return {"png": png_path, "stl": stl_path, "txt": txt_path}
