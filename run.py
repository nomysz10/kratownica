"""CLI entry point for the truss/bridge optimizer."""

import argparse

from truss import constants as C
from truss.optimize import run_optimization
from truss.report import write_report


def main():
    p = argparse.ArgumentParser(description="Optymalizacja kratownicy/mostka do płaskiego druku 3D")
    p.add_argument("--el-size", type=float, default=2.5, help="rozmiar elementu siatki (mm)")
    p.add_argument("--max-iter", type=int, default=80, help="iteracje optymalizacji SIMP")
    p.add_argument("--out", type=str, default="out", help="katalog wyjściowy")
    p.add_argument("--span", type=float, default=C.SPAN)
    p.add_argument("--mass", type=float, default=C.MAX_MASS)
    p.add_argument("--deflection", type=float, default=C.MAX_DEFLECTION)
    args = p.parse_args()

    result = run_optimization(
        span=args.span, max_mass_g=args.mass, max_deflection=args.deflection,
        d_range=C.D_RANGE, h_range=C.H_RANGE, el_size=args.el_size,
        max_iter=args.max_iter,
    )
    paths = write_report(result, args.out)
    print(f"P_max = {result.p_max:.2f} N (D={result.D:.2f}, H={result.H:.2f}, masa={result.mass_g:.2f} g)")
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
