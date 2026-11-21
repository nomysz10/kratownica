"""CLI entry point for the truss/bridge optimizer."""

import argparse
import os

import numpy as np

from truss import constants as C
from truss.mesh import make_rect_mesh
from truss.optimize import run_optimization
from truss.report import write_report
from truss.topo import topology_optimize
from truss.visualize import animate_simp


def main():
    p = argparse.ArgumentParser(description="Optymalizacja kratownicy/mostka do płaskiego druku 3D")
    p.add_argument("--el-size", type=float, default=2.5, help="rozmiar elementu siatki (mm)")
    p.add_argument("--max-iter", type=int, default=80, help="iteracje optymalizacji SIMP")
    p.add_argument("--out", type=str, default="out", help="katalog wyjściowy")
    p.add_argument("--span", type=float, default=C.SPAN)
    p.add_argument("--mass", type=float, default=C.MAX_MASS)
    p.add_argument("--deflection", type=float, default=C.MAX_DEFLECTION)
    p.add_argument("--workers", type=int, default=1, help="liczba procesów (D×H równolegle)")
    p.add_argument("--deck-load", action="store_true",
                   help="obciążenie na jezdni (pas dolny, y=0) zamiast pasa górnego")
    p.add_argument("--gif", action="store_true", help="zapisz GIF powstawania (najlepszy D,H)")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    y_load = 0.0 if args.deck_load else None

    # progress: print + best-so-far PNG after each case
    best_box = {"res": None}

    def on_case_done(res):
        if res is None:
            print("  (pominięto)")
            return
        print(f"  D={res.D:.1f} H={res.H:.1f} -> p_max={res.p_max:.1f} N, masa={res.mass_g:.1f} g")
        if best_box["res"] is None or res.p_max > best_box["res"].p_max:
            best_box["res"] = res
            try:
                write_report(res, os.path.join(args.out, "progress"))
            except Exception:
                pass

    print(f"Optymalizacja: span={args.span}, masa<={args.mass} g, ugięcie<={args.deflection} mm, "
          f"workers={args.workers}")
    result = run_optimization(
        span=args.span, max_mass_g=args.mass, max_deflection=args.deflection,
        d_range=C.D_RANGE, h_range=C.H_RANGE, el_size=args.el_size,
        max_iter=args.max_iter, n_workers=args.workers,
        keepout=None, y_load=y_load, on_case_done=on_case_done,
    )

    if result is None:
        print("Brak wykonalnego rozwiązania.")
        return

    paths = write_report(result, args.out)
    print(f"P_max = {result.p_max:.2f} N (D={result.D:.2f}, H={result.H:.2f}, masa={result.mass_g:.2f} g)")
    for k, v in paths.items():
        print(f"{k}: {v}")

    if args.gif:
        _make_gif(args, result, paths)


def _make_gif(args, result, paths):
    """Re-run SIMP for the best (D,H) collecting frames, save a GIF."""
    mesh = make_rect_mesh(args.span, result.D, args.el_size)
    volfrac = min(1.0, (args.mass / C.RHO_PLA) / (args.span * result.D * result.H))
    frames = []

    def on_iter(x_phys, _it):
        frames.append(x_phys.reshape(mesh.ny, mesh.nx).copy())

    print(f"Generowanie GIF (D={result.D}, H={result.H})...")
    topology_optimize(
        mesh, C.E, C.NU, volfrac,
        p=C.SIMP_P, rmin=max(C.FILTER_RMIN, 1.5 * args.el_size),
        max_iter=args.max_iter, move=0.2,
        keepout=None, y_load=0.0 if args.deck_load else None,
        on_iter=on_iter,
    )
    gif_path = os.path.join(args.out, "powstawanie.gif")
    animate_simp(frames, (mesh.ny, mesh.nx), args.el_size, gif_path)
    print(f"GIF: {gif_path}")


if __name__ == "__main__":
    main()
