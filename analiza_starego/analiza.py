"""Analiza starego mostka (profil z PNG) tym samym MES co nasz optymalizator."""
import numpy as np
from PIL import Image

from truss import constants as C
from truss.fea import (
    assemble_stiffness_variable,
    center_patch_load,
    recover_stress_variable,
    solve_linear,
)
from truss.mesh import make_rect_mesh
from truss.postprocess import keep_largest_component
from truss.verify import center_deflection

SPAN = 500.0
EL = 2.5
H = 8.0  # grubość w osi Z (druk płaski)

# 1. materiał = nieprzezroczyste piksele
a = np.array(Image.open("kratownicy.dxf.png").convert("RGBA"))
mat = a[:, :, 3] > 128
Hpx, Wpx = mat.shape
mm_per_px = SPAN / Wpx
height_mm = Hpx * mm_per_px

nx = int(round(SPAN / EL))
ny = int(round(height_mm / EL))
D = ny * EL
binary = np.zeros((ny, nx), dtype=bool)
for j in range(ny):
    for i in range(nx):
        px = int((i + 0.5) * EL / mm_per_px)
        py = int((height_mm - (j + 0.5) * EL) / mm_per_px)
        if 0 <= px < Wpx and 0 <= py < Hpx:
            binary[j, i] = mat[py, px]

binary = keep_largest_component(binary)
mass_g = binary.sum() * EL ** 2 * H * C.RHO_PLA
print(f"obraz {Wpx}x{Hpx} px, wysokość {height_mm:.1f} mm, siatka {nx}x{ny}")
print(f"materiał {binary.mean()*100:.1f}%, masa {mass_g:.1f} g")

# 2. podpory: najniższy węzeł z materiałem przy każdym końcu (wałki)
mesh = make_rect_mesh(SPAN, D, EL)
n_dof = 2 * mesh.nodes.shape[0]

def lowest_material_node(x_target, tol_cols):
    best = None
    for i in range(max(0, x_target - tol_cols), min(nx, x_target + tol_cols + 1)):
        rows = np.where(binary[:, i])[0]
        if len(rows):
            j = rows.min()  # najniższy wiersz z materiałem (j=0 = dół siatki)
            node = j * (nx + 1) + i
            if best is None or j < best[1]:
                best = (node, j)
    return best

left, left_row = lowest_material_node(0, 3)
right, right_row = lowest_material_node(nx - 1, 3)
print(f"podpory: lewy węzeł {left} (wiersz {left_row}), prawy węzeł {right} (wiersz {right_row})")

# fixed: ux=uy=0 na lewym, uy=0 na prawym
fixed = np.array([2 * left, 2 * left + 1, 2 * right + 1], dtype=int)

# 3. obciążenie jednostkowe na górnej krawędzi (jezdnia), 60 mm w środku
f = center_patch_load(mesh, 60.0, 1.0, y_load=mesh.ly)

E_eff = np.where(binary.ravel(), C.E, 1e-9 * C.E)
K = assemble_stiffness_variable(mesh, C.NU, H, E_eff)
u = solve_linear(K, f, fixed)

delta = center_deflection(u, mesh, SPAN / 2.0, y_load=mesh.ly)
stress = recover_stress_variable(mesh, u, C.NU, H, E_eff)
sxx, syy, sxy = stress[:, 0], stress[:, 1], stress[:, 2]
vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2)
# wyłącz osobliwość przy obciążeniu/podporach
centers = mesh.nodes[mesh.elements].mean(axis=1)
mask = (np.abs(centers[:, 0] - SPAN / 2) > 30) & (centers[:, 0] > 15) & (centers[:, 0] < SPAN - 15)
vm_peak = vm[mask].max()

p_ugiecie = 20.0 / delta
p_napr = C.SIGMA_ALLOW / vm_peak
p_max = min(p_ugiecie, p_napr)

print()
print("=== WYNIK (stary mostek, H=8 mm) ===")
print(f"masa           = {mass_g:.1f} g")
print(f"ugięcie przy 1N= {delta:.3f} mm")
print(f"  p_ugięcie    = {p_ugiecie:.1f} N  <- limit 20 mm")
print(f"  p_napręzenie = {p_napr:.1f} N  (peak vm={vm_peak:.1f} MPa)")
print(f"P_max          = {p_max:.1f} N")
print(f"N/g            = {p_max/mass_g:.2f}")
print(f"rządzi         = {'ugięcie' if p_ugiecie < p_napr else 'naprężenie'}")
