# Kratownica/mostek 3D — projekt optymalizatora

## Intencja i po co
Program, który sam generuje i optymalizuje mostek kratownicowy do **płaskiego druku 3D**,
maksymalizując obciążenie punktowe w środku przy twardych ograniczeniach. Wynik: STL gotowy do slicera
+ raport liczbowy.

## Cele / nie-cele
- **Cel:** maksymalne obciążenie P (N) w środku przęsła.
- **Cele poboczne:** organiczna geometria (nie z góry narzucona kratownica), płynnie zmienne szerokości prętów,
  stała wysokość H (płaski druk), minimum 4 mm szerokości.
- **Nie-cele:** wieloprzęsłowość (najpierw jedno przęsło; dwa równoległe potem), estetyka, produkcja masowa.
- **Anty-cele (porażka, nawet jeśli „technicznie działa"):** wynik nieprintowalny (za cienkie/zawieszone fragmenty),
  zawyżony udźwig bez weryfikacji wyboczenia, raport bez rozbicia na tryby zniszczenia.

## Ograniczenia (twarde)
| Parametr | Wartość |
|---|---|
| Rozpiętość | 500 mm |
| Ugięcie maks. | 20 mm (L/25 — konstrukcja rządzona ugięciem) |
| Masa maks. | 150 g |
| Najwęższy przekrój | 4 mm |
| Wysokość H | stała, optymalizowana (zmienna skalarna) |
| Obciążenie | punktowe w środku, rozłożone na plasterek ~15 mm |
| Materiał | PLA: E=2,8 GPa, ν=0,35, ρ=1,24 g/cm³, σ_dop=20 MPa (30 MPa / SF 1,5) |

Uwaga: 150 g / 500 mm ≈ 0,3 g/mm → średni przekrój ~242 mm² na stacji → realna głębokość D ≈ 60–120 mm.

## Podejścia (rozważone i wybór)
- **A. Kratownica parametryczna (Warren/Pratt) + optymalizacja przekrojów.** Odrzucone: topologia stała,
  suboptymalna dla obciążenia punktowego (Warren/Pratt są pod obciążenie ciągłe).
- **B. Optymalizacja topologiczna (SIMP) 2D → organiczny kształt → STL.** **Wybrane** — jedyne, które dla
  obciążenia punktowego daje teoretyczne optimum (struktura Michella).
- **C. Hybryda.** Zbyteczna: B + etap weryfikacji pokrywa potrzebę.

## Metoda (szczegóły)
1. **Siatka 2D** (plane stress, Q4) nad domeną 500×D; gęstość ρ∈[0,1], E_eff = ρ^p·E (SIMP, p=3).
   Filtrowanie gęstości (promień ~2 mm) + projekcja Heaviside'a → min. rozmiar cechy.
2. **Cel:** minimalizacja podatności (compliance) przy danej frakcji objętości
   V* = 120 970 mm³ / (500·D·H) — równoważne minimalizacji ugięcia przy stałej masie.
   Rozwiązywanie metodą OC (optimality criteria) — bez zewnętrznego solwera.
3. **Pętla zewnętrzna** po (D, H): D ∈ [40, 120] mm, H ∈ [3, 8] mm → dla każdej pary pełny przebieg.
4. **Post-processing:** próg ρ_thr → kontur → wygładzenie → morfologiczne wymuszenie min. 4 mm →
   ekstruzja o H w Z → binarny STL.
5. **Weryfikacja (finalny kształt, zagęszczona siatka):**
   - δ_jedn. = ugięcie środka przy obciążeniu 1 N → **P_ugięcie = 20 mm / δ_jedn.**
   - σ_max ≤ σ_dop → **P_naprężenie**
   - wyboczenie: szkielet 1D → siły osiowe ściskanych prętów → Euler, I_min = w·H³/12 (słaba oś,
     wyboczenie z płaszczyzny), K=1 (konserwatywnie) → **P_wyboczenie**
   - **P_max = min(P_ugięcie, P_naprężenie, P_wyboczenie)** — raport podaje wszystkie trzy + dominujący tryb.

## Decyzje, które podejmuję sam (do sprawdzenia przy czytaniu)
- Python 3 + numpy/scipy; własny MES (transparentny, bez licencji); OC zamiast MMA (wystarcza dla SIMP).
- STL pisany bezpośrednio (binarny) z ekstrudowanej siatki.
- Obciążenie na plasterku 15 mm, podpory na 15 mm (unikanie osobliwości).
- Wyboczenie szacunkiem belkowym Eulera (nie pełną analizą powłokową) — konserwatywnie i jawnie oznaczony
  jako przybliżenie; pełne LBA można dodać później.
- Wartości materiału wg tabeli; bezpieczeństwo 1,5.

## Co zostawiam wykonawcy (mnie, przy implementacji)
- Szczegóły siatki (rozmiar elementu ~2–3 mm, ~30–60k elementów), filtry, dobór ρ_thr.
- Dobór dokładnego ekstraktora konturu i morfologii (jak gwarantować 4 mm bez dziur).
- Format raportu (PNG geometrii + wykres szerokości wzdłuż ścieżki + tabela P/masa/ugięcie/H/D).
