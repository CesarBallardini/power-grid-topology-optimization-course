# Chapter 20 – Solutions

Solutions of the Lab and the Exercises of [Chapter 20](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`fault_levels.py`](src/fault_levels.py) (the hand calculation,
sequence networks and $Z_\text{bus}$), [`station_sc.py`](src/station_sc.py) (Lab Part A with pandapower),
[`rx_plane.py`](src/rx_plane.py) (Lab Part B and Exercise 5), [`case9_sc.py`](src/case9_sc.py) (Exercise 3),
[`switching_sequence.py`](src/switching_sequence.py) (Exercise 4) and [`va_diff_toop.py`](src/va_diff_toop.py)
(Exercise 6, ToOp). Each module's `main()` prints the numbers quoted below; pandapower 3.5.6 produced the
short-circuit values, ToOp `ba1874e` on CPU the angle differences. The notebooks
[examples.ipynb](src/examples.ipynb), [lab.ipynb](src/lab.ipynb) and [exercises.ipynb](src/exercises.ipynb)
import these modules. GeoGebra steps are described, not run: matplotlib or the NumPy functions of `rx_plane.py`
give the same figures and numbers.

## Lab

### Part A: short circuit with pandapower

1. **The grid.** `station_sc.station()` builds five 220 kV buses: external grids at G1 (`s_sc_max_mva=8000`,
   `s_sc_min_mva=6000`) and G2 (6 000 and 4 500 MVA), both with `rx_max=rx_min=0.1`; the station busbars S1 and
   S2 joined by `pp.create_switch(net, S1, S2, et="b", type="CB")`; lines G1–S1 (40 km), G2–S2 (60 km), S1–L and
   S2–L (30 km) and G1–G2 (150 km), all $j0.4\ \Omega/\text{km}$ with $r = 0.03\ \Omega/\text{km}$ and
   `endtemp_degree=80`; a 200 MW, 50 Mvar load at L. The arguments `extra_paths=False` and `r_ohm_per_km=0`
   reduce it to the grid of the hand calculation.

    ```python title="station_sc.py"
    --8<-- "docs/part-3-power-systems/ch20-protection-short-circuit/src/station_sc.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch20-protection-short-circuit/src){ target="_blank" } to see or download all the files of this chapter.

2. **Maximum case.** `calc_sc(net, case="max", ip=True, branch_results=False)`, coupler closed and open
   (`station_table()`):

    | Case | Coupler | S1 | S2 |
    |---|---|---|---|
    | max | closed | 10.398 kA | 10.398 kA |
    | max | open | 8.643 kA | 7.261 kA |
    | min | closed | 8.908 kA | 8.908 kA |
    | min | open | 7.456 kA | 6.311 kA |

    With the coupler closed both busbars are one node and see the same current; `skss_mw` is $\sqrt 3\,U_n I''_k$
    (3 962 MVA closed) and `ip_ka` is $\kappa\sqrt 2\,I''_k$, about 2.5 times $I''_k$ here. The coupler
    lowers the fault level of each busbar once open, by less than the hand calculation predicts: see step 3.

3. **Minimum case and the hand calculation.** The minimum case (rows `min` above) uses $c_\text{min} = 1.0$, the
   minimum feeder powers and the line resistances at the end temperature, and is 14–15 % below the maximum. The
   pandapower values with the coupler open (8.64 and 7.26 kA) are well above the chapter's 6.2 and 4.3 kA
   because the station is not radial: with the coupler open, busbar S1 is still fed by infeed 2 through
   S2–L–S1 (60 km) and through G2–G1–S1, and busbar S2 by infeed 1 through the mirror paths. Those parallel
   paths are what the hand calculation neglects. Removing them (`extra_paths=False`) and setting $r = 0$ gives
   10.428 kA closed and 6.174 / 4.254 kA open against the chapter's 10.4, 6.2 and 4.3 kA (minimum case 8.942,
   5.284 and 3.658 kA). The closed value hardly moves (10.398 against 10.428 kA) because, with both infeeds
   already on the same node, the extra paths add nothing in parallel that was not there.

4. **The recoupling angle.** `pp.runpp` with the coupler closed gives $\theta_{S1} - \theta_{S2} = 0$ (one
   node); open, with the 200 MW load fed from both sides, $0.359°$ (`recoupling_angle()`). That is the standing
   angle a synchro-check relay on the coupler would measure before reclosing: small here, because the two
   busbars stay tied through the 30 km lines and the load bus.

5. **Verdict.** With 8 kA breakers and 5 kA as the smallest current the busbar protection needs
   (`verdict()`): coupled, 10.40 kA exceeds the breaker rating, so the station cannot run coupled; split, S2 is
   acceptable (7.26 kA maximum, 6.31 kA minimum) but S1 still exceeds 8 kA (8.64 kA) because of the parallel
   paths, so the split alone does not make this station operable, and the operator would also have to open one
   of the extra paths or accept a breaker upgrade. In the chapter's radial grid the picture is the one the text
   describes: split, both busbars are below 8 kA, but S2 falls to 4.25 kA (3.66 kA in the minimum case),
   below the 5 kA the protection needs. Which constraint bites depends on the actual meshing around the
   station, which is exactly why a flow-only optimizer cannot decide it.

### Part B: distance zones in the R-X plane

1. **The polygon.** For $\theta = 30°$, $r_i = 2.5$, $r_v = 10$, $x_v = 15\ \Omega$ the vertices are $(0, 0)$,
   $(2.5, 0)$, $(2.5, 1.443)$, $(10, 5.774)$, $(10, 15)$ and $(0, 15)$ (`rx_plane.toop_polygon`): a rectangle
   $[0, r_v] \times [0, x_v]$ with its lower right corner cut off along the line through the origin at the
   angle $\theta$, except for the small foot $r_i$ along the R axis. In GeoGebra, define the four parameters
   as sliders and the polygon as `Polygon((0,0), (r_i,0), (r_i, r_i tan(θ)), (r_v, r_v tan(θ)), (r_v, x_v), (0, x_v))`;
   in matplotlib, `plt.fill(*zip(*poly))` draws the same figure.

2. **The loading path.** For a 380 kV line carrying $S = P + jQ$ the relay sees $|Z| = V^2/|S|$ at the angle
   of $S$ (`apparent_impedance`): 1 000 MVA at $\cos\varphi = 0.95$ is $(137.2, 45.1)\ \Omega$, far outside;
   the point moves towards the origin along the ray at $\varphi = 18.2°$ as the loading grows and enters the
   polygon where the ray meets its boundary, at $|Z| = 2.63\ \Omega$, i.e. at $S = 380^2 / 2.63 = 54\,872$ MVA
   (`tripping_mva`). At unity power factor the boundary is the foot $r_i = 2.5\ \Omega$ (57 760 MVA); at
   $\varphi = 60°$ the sloped edge is met at $|Z| = 17.3\ \Omega$ (8 337 MVA). Numbers of this size say that
   the example parameters of the relay table are not primary ohms of a 380 kV line: a 380 kV line loaded to
   1 000 MVA sits at $|Z| = 144\ \Omega$. Secondary ohms (after the CT and VT ratios) or a lower voltage level
   are where 10 to 15 $\Omega$ make sense.

3. **The warning factor and absolute values.** ToOp divides $|R|$ and $|X|$ by the zone factor before the test
   (`trips(poly, z, 1.41)`), which is the same as scaling the polygon by 1.41: the tripping MVA at
   $\cos\varphi = 0.95$ drops from 54 872 to 38 916 MVA (`tripping_mva(..., factor=1.41)`). Because of the
   absolute values the test is non-directional: a reverse flow of 60 000 MVA at $\cos\varphi = 0.95$ gives
   $Z = (-2.29, -0.75)\ \Omega$, a point in the third quadrant, outside the polygon as drawn, yet ToOp's
   $(|R|, |X|) = (2.29, 0.75)$ is inside and the line trips. A directional relay would not see a forward fault
   here.

4. **80° and the validity condition.** With $\theta = 80°$ the third and fourth vertices become
   $(2.5, 14.18)$ and $(10, 56.71)$: the fourth lies above $(10, 15)$, so the edge from $(10, 56.71)$ to
   $(10, 15)$ runs downwards and the sloped edge from $(2.5, 14.18)$ to $(10, 56.71)$ crosses the top edge
   from $(10, 15)$ to $(0, 15)$. `is_simple` reports the self-intersection that shapely flags as invalid. The
   polygon is simple exactly when the fourth vertex is not above the fifth, $r_v \tan\theta \le x_v$, i.e.
   $\theta \le \arctan(x_v / r_v) = \arctan 1.5 = 56.3°$ for these parameters (`is_valid`), together with
   $0 \le r_i \le r_v$. The relay table's 80° fails it, the tests' 30° passes.

    ```python title="rx_plane.py"
    --8<-- "docs/part-3-power-systems/ch20-protection-short-circuit/src/rx_plane.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch20-protection-short-circuit/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

For a fault on phase a through zero impedance, the boundary conditions are $I_b = I_c = 0$ and $V_a = 0$.
Transforming the currents, $I_0 = I_1 = I_2 = \tfrac13 (I_a + 0 + 0) = I_a / 3$: the three sequence currents are
equal. The sequence voltages at the fault are $V_1 = E - Z_1 I_1$, $V_2 = -Z_2 I_2$, $V_0 = -Z_0 I_0$, and
$V_a = V_0 + V_1 + V_2 = 0$ gives $E - (Z_1 + Z_2 + Z_0)\,I_1 = 0$, so

$$
I_1 = \frac{E}{Z_1 + Z_2 + Z_0}, \qquad I_f = I_a = 3 I_1 = \frac{3E}{Z_1 + Z_2 + Z_0},
$$

the three sequence networks in series. The three-phase fault is $I_f^{(3)} = E / Z_1$, so

$$
\frac{I_f^{(1)}}{I_f^{(3)}} = \frac{3 Z_1}{Z_1 + Z_2 + Z_0} \approx \frac{3 Z_1}{2 Z_1 + Z_0} \quad (Z_2 \approx Z_1),
$$

which exceeds one exactly when $Z_0 < Z_1$ (`fault_levels.slg_current`: 1.2 for $Z_0 = Z_1/2$, 1.0 for
$Z_0 = Z_1$, 0.6 for $Z_0 = 3 Z_1$). Lines have $Z_0 \approx 3 Z_1$, so far from the sources the single-phase
current is the smaller one. At a solidly grounded transformer the zero-sequence path is short: a Yg–Δ winding
lets zero-sequence current circulate in the delta and presents only its leakage impedance to ground, and a
generator's $Z_0$ is also small, so near such transformers $Z_0 < Z_1$ and the single line-to-ground current
exceeds the three-phase one. That is why breaker ratings and earthing designs are checked against the
single-phase fault as well, and why a busbar split that changes $Z_1$ at a station changes the two fault types
by different amounts.

```python title="fault_levels.py"
--8<-- "docs/part-3-power-systems/ch20-protection-short-circuit/src/fault_levels.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch20-protection-short-circuit/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

`fault_levels.FIVE_BUS` is a 110 kV grid with reactances in ohm: branches 1–2 (8), 2–3 (6), 3–4 (6), 4–5 (8),
2–4 (10) and 1–3 (12), and source impedances of 2 $\Omega$ at bus 1 and 3 $\Omega$ at bus 5 (1-based bus
numbers). $Y_\text{bus}$ holds the branch admittances plus the source admittances on the diagonal, and
$Z_\text{bus} = Y_\text{bus}^{-1}$; the diagonal entry $Z_{kk}$ is the Thevenin impedance at bus $k$ and
$I''_k = 1.1 \cdot 110 / (\sqrt 3\, Z_{kk})$. Before the split: 38.47, 12.84, 12.37, 12.86 and 27.02 kA at
buses 1 to 5. Splitting bus 3 as in Chapter 18, branches 3–4 and 1–3 moving to a new node 6 and branch 2–3
staying: 38.42, 10.61, 5.55, 12.70, 26.96 kA and 9.85 kA at the new node. Changes: −0.1 %, −17.4 %, −55.1 %,
−1.2 %, −0.2 %: buses 2 and 3 change by more than 10 % (`split_comparison`). Bus 3 loses the parallel
contribution of infeed 1 through 1–3 (its half of the old station now hangs on 2–3 alone), and bus 2 loses
the path 2–3–4 towards infeed 2. Buses 1 and 5 barely move because their $Z_{kk}$ is dominated by the source
impedance right at the bus (2 and 3 $\Omega$) and the network beyond contributes only a parallel branch of
tens of ohms; a change deep in the mesh alters that parallel branch by a little, and a little of a small
contribution is nothing. In $Z_\text{bus}$ terms: a split is a low-rank change of $Y_\text{bus}$, and the
resulting change of $Z_{kk}$ is proportional to the square of the transfer impedance between bus $k$ and the
split station, which decays with electrical distance. Fault levels are a local property, which is why a split
to limit them works, and why it does not help a remote station.

## Exercise 3

`case9_sc.with_sc_data()` adds what `calc_sc` refuses to run without: `s_sc_max_mva`, `s_sc_min_mva`, `rx_max`
and `rx_min` on the external grid (5 000 and 3 000 MVA, 0.1), `vn_kv`, `sn_mva`, `xdss_pu`, `rdss_ohm` and
`cos_phi` on the two generators (345 kV, 250 MVA, 0.2, 0.05 $\Omega$, 0.85) and `endtemp_degree` on the lines
(80 °C); without the latter `case="min"` stops with "Specify end temperature for lines". `calc_sc` also warns
that peak and thermal currents are only implemented for faults far from generators, which is a limitation to
remember near buses 2 and 3. The results (`max_min_table`), $I''_k$ in kA:

| Bus | max | min | min/max |
|---|---|---|---|
| 1 | 9.087 | 5.673 | 0.624 |
| 2 | 3.148 | 2.849 | 0.905 |
| 3 | 3.146 | 2.846 | 0.905 |
| 4 | 3.239 | 2.678 | 0.827 |
| 5 | 1.794 | 1.573 | 0.877 |
| 6 | 2.464 | 2.215 | 0.899 |
| 7 | 1.931 | 1.739 | 0.901 |
| 8 | 2.461 | 2.212 | 0.899 |
| 9 | 1.868 | 1.637 | 0.877 |

Three things separate the two cases. The voltage factor: $c_\text{max} = 1.1$ in the maximum case and
$c_\text{min} = 1.0$ in the minimum case (IEC 60909 at this voltage level), a flat 10 % that every bus shows:
at the generator buses 2 and 3 the ratio is 0.905 and almost nothing else changes, since the generator
impedances are the same in both cases. The feeder: at bus 1 the external grid drops from 5 000 to 3 000 MVA,
and the ratio is 0.624. The line temperature: in the minimum case the line resistances are taken at the end
temperature, $R(\vartheta) = R_{20}\,[1 + 0.004\,(\vartheta - 20)]$, which raises the path impedance a little;
at bus 5, in the middle of the long lines, the minimum current is 1.580, 1.573 and 1.562 kA for 20, 80 and
160 °C (`endtemp_effect`), a change below 2 % because these 345 kV lines are mostly reactance. The maximum
case sizes the equipment (breaker ratings, with cold conductors and strong feeders), the minimum case sets the
protection (the smallest current a relay must still detect, with hot conductors and weak feeders), and both
limits must hold at all times (SO GL Art. 30–31).

```python title="case9_sc.py"
--8<-- "docs/part-3-power-systems/ch20-protection-short-circuit/src/case9_sc.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch20-protection-short-circuit/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

Station: busbars BB1 and BB2; the line bay with its breaker CB, busbar disconnectors Q1 (to BB1) and Q2 (to
BB2) and line disconnector Q9; the coupler bay with its breaker CBC and disconnectors QC1 (to BB1) and QC3 (to
BB2). Start: Q1, CB and Q9 closed (the line is on BB1), everything in the coupler bay open. A disconnector may
only operate when it carries no current: either its two ends are already tied by another closed path (then no
voltage stands across it and the current commutates), or one end is dead. A breaker may switch under load.

| Step | Device | Why it is allowed | Line on both busbars? |
|---|---|---|---|
| 1. close QC1 | disconnector | coupler bay dead | no |
| 2. close QC3 | disconnector | coupler bay dead | no |
| 3. close CBC | **breaker** | breakers switch under load; BB1 and BB2 are now paralleled | yes (through the coupler) |
| 4. close Q2 | disconnector | parallel path BAY–Q1–BB1–coupler–BB2: no voltage across Q2 | yes (Q1 and Q2 both closed) |
| 5. open Q1 | disconnector | parallel path through Q2 and the coupler: its current commutates | yes, until it opens |
| 6. open CBC | **breaker** | interrupts the coupler current: the station is split | no |
| 7. open QC1 | disconnector | coupler bay dead | no |
| 8. open QC3 | disconnector | coupler bay dead | no |

Steps 4 and 5 are the make-before-break reassignment: between them the bay is connected to both busbars at
once through its own disconnectors, which is only harmless because the coupler holds the busbars at the same
voltage; the line is in parallel across both busbars from step 3 to step 5. `switching_sequence.run` checks
every step with those rules and refuses the two classic mistakes: closing Q2 with the coupler open ("would make
or break load current", the two busbars are live at different voltages) and opening Q1 with the line on load
and no parallel path. Moving a bay in a station that is *already* split needs steps 1–3 first, i.e. a temporary
recoupling, which may be impossible because of the loop flow it creates or because the coupled fault level
exceeds a breaker rating (Lab Part A, step 5). ToOp's `switching_distance` counts how many bays change busbar,
not whether this sequence exists for them.

```python title="switching_sequence.py"
--8<-- "docs/part-3-power-systems/ch20-protection-short-circuit/src/switching_sequence.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch20-protection-short-circuit/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

The line is $Z_L = 100 \cdot (0.03 + j0.3) = 3 + j30\ \Omega$. Zone 1 reaches 80 %, $X = 24\ \Omega$, zone 2
120 %, $X = 36\ \Omega$; both quadrilaterals are drawn with a resistive blinder at $R = 10\ \Omega$ and the
reactance line as their top edge (`rx_plane.quadrilateral_zone`; in GeoGebra, `Polygon((0,0), (10,0), (10,24),
(0,24))` and the same with 36, plus the line impedance as a segment from the origin to $(3, 30)$). A fault at
90 % of the line with $R_f = 5\ \Omega$ seen from the near end, with all the fault current supplied through the
relay, is $Z = 0.9\,Z_L + R_f = (7.7, 27.0)\ \Omega$: beyond zone 1, inside zone 2, as it should be for a fault
in the last 20 % of the line. Now let the split busbar at the remote end feed the fault as well, with half
the fault current: the relay's current is half the fault current, so the fault resistance appears doubled,
$Z = 0.9\,Z_L + 2 R_f = (12.7, 27.0)\ \Omega$ (`infeed_apparent_impedance`). The point moves to the right by
$5\ \Omega$, past the resistive blinder of both zones: the relay under-reaches, it sees an impedance larger
than the fault position, and neither zone 1 nor zone 2 picks up; clearing is left to the backup zone or the
remote end. Infeed always makes a distance relay under-reach (the apparent impedance grows); zone 1 is set to
80 % precisely so that errors of this kind do not make it over-reach into the next line. A topology change
that adds an infeed behind a relay therefore changes the apparent fault position, and settings validated for
the unsplit station may not protect the split one.

## Exercise 6

`va_diff_toop.py` runs the Chapter 19 AC N-1 (nine single-line outages on `grid.xiidm`) through the Chapter 19
module and reads `va_diff_results`: nine rows, one per outaged branch, each with the angle across the open
ends of the outaged line (the grid has no open retained switches, so no switch rows). None exceeds 20° or
10°; the largest are 0.485° for the outages of L1 and L2, 0.424° for L8 and L3, 0.179° for L4 to L7, and the
L9 row is NaN because that outage leaves one end without voltage (no angle to compare). The worst case,
reclosing L1 after its outage: an operator (or the automatic reclosing scheme) would check that the fault is
cleared and the protection has reset, that both ends are live and within their voltage bands, that the
standing angle across the open breaker (0.49° here, far below a typical synchro-check setting of 10–20° and
below ToOp's `critical_va_diff_degree = 20`) and the slip are within limits, and then close one end first and
the other with synchro-check supervision. L9 is different: with one end dead there is nothing to synchronise;
it is energised from the live end ("dead line, live bus" closing) and the remote breaker is closed afterwards
under synchro-check. The metric does its job: a large standing angle after a topology change is what would
turn a routine reclosing into a torque shock on nearby machines, and that is what `max_va_diff_n_1` flags.

```python title="va_diff_toop.py"
--8<-- "docs/part-3-power-systems/ch20-protection-short-circuit/src/va_diff_toop.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch20-protection-short-circuit/src){ target="_blank" } to see or download all the files of this chapter.
