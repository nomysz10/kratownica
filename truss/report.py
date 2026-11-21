"""Report writer: geometry PNG, stress/buckling PNGs, extruded STL, text summary."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from truss.constants import SAFETY_FACTOR
from truss.stl import extrude_binary_to_triangles, write_binary_stl
from truss.visualize import plot_buckling, plot_stress


def write_report(result, outdir):
    """Write PNG/STL/txt for a Result. Returns dict of paths."""
    os.makedirs(outdir, exist_ok=True)
    png_path = os.path.join(outdir, "geometria.png")
    stress_x1 = os.path.join(outdir, "naprezenia_x1.png")
    stress_x10 = os.path.join(outdir, "naprezenia_x10.png")
    buckl_x1 = os.path.join(outdir, "wyboczenie_x1.png")
    buckl_x10 = os.path.join(outdir, "wyboczenie_x10.png")
    stl_path = os.path.join(outdir, "mostek.stl")
    txt_path = os.path.join(outdir, "raport.txt")

    # geometry PNG
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(result.binary.astype(float), cmap="gray_r", origin="lower",
              extent=[0, result.binary.shape[1] * result.el_size,
                      0, result.binary.shape[0] * result.el_size])
    ax.set_xlabel("span [mm]")
    ax.set_ylabel("depth [mm]")
    ax.set_title(f"mostek D={result.D:.1f} H={result.H:.1f}")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)

    # stress + buckling at real scale (x1) and exaggerated (x10)
    p_max = result.p_max
    plot_stress(stress_x1, result.mesh, result.binary, result.u, result.vm,
                scale=p_max, title=f"von Mises przy {p_max:.0f} N (x1)")
    plot_stress(stress_x10, result.mesh, result.binary, result.u, result.vm,
                scale=10 * p_max, title="von Mises (x10)")
    plot_buckling(buckl_x1, result.mesh, result.members, result.u,
                  scale=p_max, p_max=p_max, sf=SAFETY_FACTOR, title="wykorzystanie wyboczeniowe (x1)")
    plot_buckling(buckl_x10, result.mesh, result.members, result.u,
                  scale=10 * p_max, p_max=p_max, sf=SAFETY_FACTOR, title="wykorzystanie wyboczeniowe (x10)")

    # STL: voxel-extrude the smoothed material region (holes preserved)
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

    return {
        "png": png_path,
        "stress_x1": stress_x1,
        "stress_x10": stress_x10,
        "buckl_x1": buckl_x1,
        "buckl_x10": buckl_x10,
        "stl": stl_path,
        "txt": txt_path,
    }
