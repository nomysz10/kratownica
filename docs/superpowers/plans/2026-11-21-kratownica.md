# Kratownica/mostek 3D — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Program w Pythonie, który optymalizacją topologiczną (SIMP) generuje organiczny mostek na płaski druk 3D, maksymalizuje obciążenie punktowe w środku przy ograniczeniach (ugiêcie ≤ 20 mm, masa ≤ 150 g, min. 4 mm) i zapisuje STL + raport.

**Architecture:** 2D plane-stress FEM (Q4) na regularnej siatce kwadratowej + optymalizacja SIMP metodą OC. Wynik (pole gęstości) progowany, wygładzany, wymuszane min. 4 mm, szkieletowany, weryfikowany (ugiêcie/naprê¿enie/wyboczenie Eulera) i ekstrudowany o H do binarnego STL. Pętla zewnętrzna po (D, H) wybiera optimum.

**Tech Stack:** Python 3, numpy, scipy, scikit-image, matplotlib, pytest. Jednostki wewnętrzne: mm, N, MPa.

**Spec:** `../specs/2026-11-21-kratownica-design.md` (plik projektu; w tym repo leży w katalogu głównym jako `2026-11-21-kratownica-design.md`)

## Global Constraints

- Jednostki: długość mm, siła N, naprężenie MPa, moduł MPa, masa g, objętość mm³.
- PLA: `E = 2800.0`, `NU = 0.35`, `RHO_PLA = 1.24e-3` (g/mm³), `SIGMA_ALLOW = 20.0` (MPa), `SAFETY_FACTOR = 1.5`.
- Spec: `SPAN = 500.0`, `MAX_DEFLECTION = 20.0`, `MAX_MASS = 150.0`, `MIN_WIDTH = 4.0`, `LOAD_PATCH = 15.0`, `SUPPORT_PATCH = 15.0`, `D_RANGE = (40.0, 120.0)`, `H_RANGE = (3.0, 8.0)`.
- SIMP: `SIMP_P = 3.0`, `FILTER_RMIN = 2.0` (mm), metoda OC, move limit 0.2.
- Siatka: elementy kwadratowe; `make_rect_mesh(lx, ly, el_size)` wymaga `lx/el_size` i `ly/el_size` całkowite (inaczej `ValueError`).
- `volfrac = min(1.0, 120970.0 / (SPAN * D * H))` — domena mniejsza niż objętość materiału = blok pełny.
- Wszystkie stałe w `truss/constants.py`, importowane — nigdy na sztywno w modułach.

## Review Focus

1. **Pomieszane jednostki** (mm vs m, MPa vs Pa) → P_max błędne o rzędy wielkości. Pinowane testami z wartościami bezwzględnymi (Eulera 147.39 N, ugiêcie EB) w Task 2 i 7.
2. **Masa > 150 g po progowaniu** (SIMP trzyma frakcję, ale próg + morfologia zmieniają masę). Task 8 asercja `mass <= MAX_MASS`.
3. **Min. 4 mm nie wymuszone po morfologii** na cienkich prętach. Test `min_feature_width` na 2 mm pręcie → ≥ 4 mm w Task 5.
4. **Odłączone/floating wyspy materiału** po progowaniu (niedrukowalne). Test `keep_largest_component` w Task 5.
5. **Wyboczenie o złą oś** (I o mocnej osi → niezachowawcze). Test Eulera z `I = w*H**3/12` (słaba oś, H < w) w Task 7.

---

### Task 1: Szablon projektu + stałe + siatka

**Files:**
- Create: `requirements.txt`
- Create: `truss/__init__.py`
- Create: `truss/constants.py`
- Create: `truss/mesh.py`
- Test: `tests/test_mesh.py`

**Interfaces:**
- Produces: `truss.constants` (wszystkie stałe z Global Constraints jako floaty); `Mesh` (dataclass: `nodes: np.ndarray` (N,2) mm, `elements: np.ndarray` (Nel,4) int Q4, `nx: int`, `ny: int`, `lx: float`, `ly: float`, `el_size: float`); `make_rect_mesh(lx: float, ly: float, el_size: float) -> Mesh`.

- [ ] **Step 1: Napisz testy siatki**

```python
def test_mesh_sizes():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    assert m.nx == 80 and m.ny == 4
    assert m.nodes.shape == ((81 * 5), 2)
    assert m.elements.shape == (320, 4)

def test_mesh_origin_and_span():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    assert m.nodes[:, 0].min() == 0.0 and m.nodes[:, 0].max() == 200.0
    assert m.nodes[:, 1].min() == 0.0 and m.nodes[:, 1].max() == 10.0

def test_mesh_rejects_non_square():
    import pytest
    with pytest.raises(ValueError):
        make_rect_mesh(200.0, 11.0, 2.5)
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_mesh.py -v`
Expected: FAIL (ImportError/brak `make_rect_mesh`).

- [ ] **Step 3: Zaimplementuj `make_rect_mesh` w `truss/mesh.py` + stałe w `truss/constants.py`**

Węzły w porządku row-major (x rośnie szybciej), elementy Q4 przeciwnie do zegara. Funkcja podnosi `ValueError`, gdy `lx % el_size != 0` lub `ly % el_size != 0`.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_mesh.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git init  # jeśli nie ma repozytorium
git add requirements.txt truss/ tests/test_mesh.py
git commit -m "feat: scaffold, constants, rectangular mesh"
```

---

### Task 2: Plane-stress FEM

**Files:**
- Create: `truss/fea.py`
- Test: `tests/test_fea.py`

**Interfaces:**
- Consumes: `Mesh`, `make_rect_mesh` (Task 1).
- Produces: `assemble_stiffness(mesh: Mesh, E: float, nu: float, thickness: float) -> scipy.sparse.csr_matrix`; `solve_linear(K, f: np.ndarray, fixed_dofs: np.ndarray) -> np.ndarray` (u, 2N); `compliance(u: np.ndarray, f: np.ndarray) -> float`; `recover_von_mises(mesh, u, E, nu, thickness) -> np.ndarray` (Nel,).

- [ ] **Step 1: Napisz test — wspornik vs Euler-Bernoulli**

```python
def test_cantilever_tip_deflection():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    K = assemble_stiffness(m, 2800.0, 0.35, 1.0)
    # utwierdzenie: wszystkie węzły x=0 (węzły 0..4) — ux=2n, uy=2n+1
    fixed = np.array([d for n in range(5) for d in (2*n, 2*n + 1)])
    f = np.zeros(2 * m.nodes.shape[0])
    # row-major: idx = row*(nx+1)+col; tip (col=80, row=2) = 2*81+80
    tip_node = 2 * (m.nx + 1) + m.nx
    f[2 * tip_node + 1] = -10.0
    u = solve_linear(K, f, fixed)
    assert abs(u[2 * tip_node + 1] + 114.2857) < 6.0  # EB: PL^3/(3EI), I=h^3*t/12
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_fea.py -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj `truss/fea.py`**

Element Q4 plane-stress (biliniowy), całkowanie 2×2 Gaussa. `assemble_stiffness` zwraca CSR (indeksy DOF: `2*n + d`, d∈{0,1} = x,y). `solve_linear` przez `scipy.sparse.linalg.spsolve` z zerowaniem rzędów DOF odebranych. `recover_von_mises`: σ w punktach Gaussa → ekstrapolacja do środka elementu → von Mises.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_fea.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/fea.py tests/test_fea.py
git commit -m "feat: plane-stress Q4 FEM"
```

---

### Task 3: Filtr gęstości + projekcja Heaviside'a

**Files:**
- Create: `truss/topo.py` (tylko filtr + projekcja)
- Test: `tests/test_topo.py`

**Interfaces:**
- Consumes: `Mesh` (Task 1).
- Produces: `density_filter(x: np.ndarray, mesh: Mesh, rmin: float) -> np.ndarray`; `heaviside_projection(x: np.ndarray, beta: float, eta: float) -> np.ndarray`.

- [ ] **Step 1: Napisz testy**

```python
def test_filter_preserves_volume():
    rng = np.random.default_rng(0)
    m = make_rect_mesh(120.0, 40.0, 2.0)
    x = rng.random(m.elements.shape[0])
    y = density_filter(x, m, rmin=2.0)
    assert 0.0 <= y.min() and y.max() <= 1.0
    assert abs(y.sum() - x.sum()) < 0.02 * x.sum()

def test_projection_range():
    rng = np.random.default_rng(1)
    x = rng.random(100)
    y = heaviside_projection(x, beta=8.0, eta=0.5)
    assert 0.0 <= y.min() and y.max() <= 1.0
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_topo.py -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj filtr i projekcję**

Filtr (standardowy, stożkowy): `w_ij = max(0, rmin - dist(i,j))`, `y_i = Σ w_ij x_j / Σ w_ij` po środkach elementów. Projekcja: `y = (tanh(beta*eta) + tanh(beta*(x-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))`.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_topo.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/topo.py tests/test_topo.py
git commit -m "feat: density filter and Heaviside projection"
```

---

### Task 4: Optymalizacja topologiczna SIMP + OC

**Files:**
- Modify: `truss/topo.py`
- Test: `tests/test_topo.py`

**Interfaces:**
- Consumes: `assemble_stiffness`, `solve_linear`, `compliance` (Task 2); `density_filter`, `heaviside_projection` (Task 3).
- Produces: `topology_optimize(mesh, E, nu, volfrac, *, p, rmin, max_iter, move) -> tuple[np.ndarray, np.ndarray]` — `(rho, compliance_history)`.

- [ ] **Step 1: Napisz test — wspornik SIMP**

```python
def test_topo_cantilever_reduces_compliance():
    m = make_rect_mesh(120.0, 40.0, 2.0)
    rho, hist = topology_optimize(m, 2800.0, 0.35, volfrac=0.5,
                                  p=3.0, rmin=1.5, max_iter=60, move=0.2)
    assert hist[-1] < hist[0] * 0.8          # podatność spada
    assert abs(rho.mean() - 0.5) < 0.01      # frakcja utrzymana
    assert np.all(np.isfinite(rho)) and 0.0 <= rho.min() and rho.max() <= 1.0
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_topo.py::test_topo_cantilever_reduces_compliance -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj `topology_optimize`**

SIMP `E_eff = E_min + x^p (E - E_min)` (E_min = 1e-9·E). Wrażliwości: `dC/dx_e = -p x_e^{p-1} u_e^T K_e u_e`, po filtrze/projekcji regułą łańcuchową. Metoda OC: mnożnik Lagrange'a bisekcją do spełnienia `volfrac`; aktualizacja z move limit 0.2. Obciążenie: jednostkowa siła pionowa w środku górnej krawędzi, podpory na obu dolnych rogach (lewy: ux=uy=0, prawy: uy=0). Zwróć `compliance` po każdej iteracji.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_topo.py::test_topo_cantilever_reduces_compliance -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/topo.py tests/test_topo.py
git commit -m "feat: SIMP topology optimization with OC"
```

---

### Task 5: Post-processing (próg, kontur, min. szerokość, szkielet)

**Files:**
- Create: `truss/postprocess.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: `make_rect_mesh` (Task 1).
- Produces: `threshold_binary(rho: np.ndarray, thr: float) -> np.ndarray` (bool, (ny,nx), element-centered); `keep_largest_component(binary) -> np.ndarray`; `enforce_min_width(binary, pixel_size: float, min_mm: float) -> np.ndarray`; `min_feature_width(binary, pixel_size: float) -> float`; `extract_contours(binary, pixel_size: float) -> list[np.ndarray]` (polygony (N,2) mm); `Member(points: np.ndarray, width: float, length: float)`; `skeletonize_members(binary, pixel_size: float) -> list[Member]`.

- [ ] **Step 1: Napisz testy**

```python
def test_min_width_enforced():
    img = np.zeros((100, 100), dtype=bool)
    img[:, 49:51] = True            # pręt 2 mm (pixel = 1 mm)
    out = enforce_min_width(img, pixel_size=1.0, min_mm=4.0)
    assert min_feature_width(out, 1.0) >= 4.0

def test_largest_component():
    img = np.zeros((50, 50), dtype=bool)
    img[10:40, 10:40] = True        # główny blok
    img[0:3, 0:3] = True            # wyspa odłączona
    out = keep_largest_component(img)
    assert out[0:3, 0:3].sum() == 0
    assert out[10:40, 10:40].all()

def test_skeleton_member():
    img = np.zeros((30, 70), dtype=bool)
    img[10:20, 5:65] = True         # prostokąt 60×10 mm
    members = skeletonize_members(img, pixel_size=1.0)
    assert len(members) <= 3
    assert any(abs(m.length - 60.0) < 6.0 and 8.0 < m.width < 12.0 for m in members)
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj `truss/postprocess.py`**

Użyj `skimage.morphology` (skeletonize/medial_axis), `skimage.morphology.binary_dilation`, `skimage.measure.find_contours`, `skimage.measure.label` + `regionprops`, `scipy.ndimage.distance_transform_edt`. `enforce_min_width` = dylatacja/domknięcie tak, by `min_feature_width = 2·max(distance_transform po szkielecie)` ≥ min_mm. `skeletonize_members`: szkielet → podział na segmenty (punkty końcowe/rozgałęzienia) → `length` = długość polilinii, `width` = 2·średnia odległość od brzegu wzdłuż osi.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_postprocess.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/postprocess.py tests/test_postprocess.py
git commit -m "feat: threshold, contour, min-width, skeleton post-processing"
```

---

### Task 6: Zapis STL

**Files:**
- Create: `truss/stl.py`
- Test: `tests/test_stl.py`

**Interfaces:**
- Consumes: kontury z Task 5.
- Produces: `extrude_contour_to_triangles(polygon: np.ndarray, height: float) -> np.ndarray` ((N,3,3) wierzchołki mm); `write_binary_stl(path: str, triangles: np.ndarray) -> None`.

- [ ] **Step 1: Napisz testy**

```python
def test_extrude_rectangle_triangle_count_and_z():
    poly = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=float)
    tris = extrude_contour_to_triangles(poly, height=4.0)
    assert tris.shape == (12, 3, 3)          # 4 ściany + 2 pokrywy
    assert set(np.round(tris[:, :, 2].ravel(), 6)) <= {0.0, 4.0}

def test_binary_stl_roundtrip(tmp_path):
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)
    p = tmp_path / "a.stl"
    write_binary_stl(str(p), tri)
    data = p.read_bytes()
    assert data[:80] == b" " * 80          # nagłówek
    assert int.from_bytes(data[80:84], "little") == 1
    assert len(data) == 84 + 50 * 1
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_stl.py -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj `truss/stl.py`**

Ekstruzja: dolna pokrywa (triangulacja wielokąta wentylatorem), górna (przesunięta o H), ściany boczne (każdy segment = 2 trójkąty). Normalne liczone z iloczynu wektorowego, skierowane na zewnątrz. Binarny STL: 80-bajtowy nagłówek + uint32 liczba + rekordy 50 B (normalna float32×3 + wierzchołki float32×9 + atrybut uint16).

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_stl.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/stl.py tests/test_stl.py
git commit -m "feat: binary STL writer with contour extrusion"
```

---

### Task 7: Weryfikacja (ugiêcie, naprê¿enie, wyboczenie)

**Files:**
- Create: `truss/verify.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Consumes: `assemble_stiffness`, `solve_linear`, `recover_von_mises` (Task 2); `skeletonize_members` (Task 5).
- Produces: `simulate_binary_shape(mesh, binary, E, nu, H, load: float) -> tuple[np.ndarray, np.ndarray]` (u, von Mises); `center_deflection(u, mesh, center_x: float) -> float`; `euler_critical_load(width, height, length, E, K: float = 1.0) -> float`; `member_axial_force(axis: np.ndarray, stress: np.ndarray, width, height) -> float`; `evaluate_shape(mesh, binary, pixel_size, E, nu, H, sigma_allow, sf, max_deflection) -> dict` (klucze: `mass_g`, `deflection_at_1N`, `p_ugiecie`, `p_naprezenie`, `p_wyboczenie`, `p_max`, `breakdown`).

- [ ] **Step 1: Napisz testy**

```python
def test_euler_weak_axis():
    # w=10, H=4, L=100, E=2800: I = w*H^3/12 = 53.333 -> Pcr = 147.39 N
    pcr = euler_critical_load(width=10.0, height=4.0, length=100.0, E=2800.0)
    assert abs(pcr - 147.39) < 0.5

def test_member_axial_force():
    n = member_axial_force(axis=np.array([1.0, 0.0]),
                           stress=np.array([10.0, 0.0, 0.0]),  # sxx=10 MPa
                           width=10.0, height=4.0)
    assert abs(n - 400.0) < 1e-9

def test_solid_beam_load_limit():
    m = make_rect_mesh(200.0, 10.0, 2.5)
    binary = np.ones((m.ny, m.nx), dtype=bool)
    r = evaluate_shape(m, binary, 2.5, 2800.0, 0.35, 1.0, 20.0, 1.5, 20.0)
    # EB: δ(1N) = L^3/(48EI) = 0.7143 mm -> p_ugiecie = 20/0.7143 = 28.0 N
    assert abs(r["p_ugiecie"] - 28.0) < 1.7
    assert abs(r["mass_g"] - 200*10*1*1.24e-3) < 0.01
    assert r["p_max"] == min(r["p_ugiecie"], r["p_naprezenie"], r["p_wyboczenie"])
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_verify.py -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj `truss/verify.py`**

`simulate_binary_shape`: `E_eff = (binary ? E : 1e-9*E)` na siatce, obciążenie jednostkowe rozłożone na węzły górnej krawędzi w pasie ±LOAD_PATCH/2 wokół środka, podpory jak w Task 4. `p_ugiecie = max_deflection / deflection_at_1N`. `p_naprezenie = sigma_allow * (load / von_mises_max_at_load)` liniowo. Wyboczenie: dla każdego pręta szkieletu o `axis` i polu w·H, siła osiowa przy obciążeniu jednostkowym `N_unit` (z pola naprężeń w środku pręta); pręty ściskane (N_unit < 0) → `p_wyb = euler_critical_load / (sf * |N_unit|)`; `p_wyboczenie = min` po ściskanych. `p_max = min` z trzech, `breakdown` nazywa dominujący tryb.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_verify.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/verify.py tests/test_verify.py
git commit -m "feat: deflection, stress, Euler buckling verification"
```

---

### Task 8: Pętla optymalizacji + raport + CLI

**Files:**
- Create: `truss/optimize.py`
- Create: `truss/report.py`
- Create: `run.py`
- Test: `tests/test_optimize.py`

**Interfaces:**
- Consumes: `topology_optimize` (Task 4), postprocess (Task 5), `extrude_contour_to_triangles`/`write_binary_stl` (Task 6), `evaluate_shape` (Task 7).
- Produces: `Result` (dataclass: `D, H, rho, binary, p_max, mass_g, deflection_at_1N, breakdown, members, compliance_history`); `run_optimization(span, max_mass_g, max_deflection, d_range, h_range, el_size) -> Result`; `report.write_report(result, outdir) -> dict[str, str]` (ścieżki PNG/STL/txt); `run.py` (argparse: `--el-size`, `--out`, `--max-iter`).

- [ ] **Step 1: Napisz test — smoke test całego potoku**

```python
def test_run_optimization_smoke(tmp_path):
    r = run_optimization(span=100.0, max_mass_g=6.0, max_deflection=20.0,
                         d_range=(20.0, 30.0), h_range=(3.0, 4.0), el_size=2.0)
    assert r.p_max > 0 and np.isfinite(r.p_max)
    assert r.mass_g <= 6.0 + 1e-6
    paths = write_report(r, str(tmp_path))
    assert os.path.exists(paths["stl"])
    assert os.path.getsize(paths["stl"]) > 0
    assert os.path.exists(paths["png"])
    assert "p_max" in open(paths["txt"]).read()
```

- [ ] **Step 2: Uruchom, potwierdź porażkę**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: FAIL.

- [ ] **Step 3: Zaimplementuj `run_optimization`, `write_report`, `run.py`**

Pętla po (D, H) z `D_RANGE`/`H_RANGE` (lub podanymi): `volfrac = min(1.0, 120970/(span*D*H))` (dla testu przeskaluj: `max_mass_g/RHO_PLA`), `topology_optimize` → próg → `keep_largest_component` → `enforce_min_width(MIN_WIDTH)` → `evaluate_shape` → zapisz `Result` z najlepszym `p_max`. `write_report`: PNG geometrii (matplotlib, contourf z `rho`), PNG rozkładu szerokości prętów, STL (`extrude_contour_to_triangles` na największym konturze × H), txt z tabelą `D, H, mass_g, p_max, breakdown`. `run.py` wywołuje z argumentami CLI.

- [ ] **Step 4: Uruchom, potwierdź przejście**

Run: `python -m pytest tests/test_optimize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add truss/optimize.py truss/report.py run.py tests/test_optimize.py
git commit -m "feat: optimization driver, report, CLI"
```

---

### Task 9: Pełny przebieg docelowy (bez testu — walidacja wyniku)

**Files:** brak nowych (uruchomienie `run.py`).

**Interfaces:** consumes wszystko wyżej.

- [ ] **Step 1: Uruchom pełną optymalizację docelową**

Run: `python run.py --el-size 2.5 --max-iter 80 --out out/`
Expected: kończy się bez błędu; `out/report.txt` zawiera `p_max`, `mass_g`, `p_ugiecie`, `p_naprezenie`, `p_wyboczenie`, `D`, `H`; `out/mostek.stl` > 0 bajtów.

- [ ] **Step 2: Otwórz PNG i sprawdź sensowność geometryczną**

Expected: organiczny wachlarz/łuk bez odłączonych wysp, pręty ≥ 4 mm, podpory połączone z punktem obciążenia.

- [ ] **Step 3: Otwórz STL w slicerze, sprawdź czy wodoszczelny i płaski**

Expected: slicer akceptuje plik; górna/dolna powierzchnia w jednej płaszczyźnie Z (stała H).

- [ ] **Step 4: Commit wynikowych plików (opcjonalnie)**

```bash
git add out/ ; git commit -m "chore: add reference optimization output"
```
