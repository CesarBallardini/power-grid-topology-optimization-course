# Chapter 20. Protection, short circuit and switching feasibility

**Weeks:** 2 · **Semester:** 4 · **Prerequisites:** [Chapter 10](../part-2-circuits/ch10-three-phase-per-unit.md), [Chapter 15](ch15-ac-power-flow.md), [Chapter 18](ch18-sensitivity-factors-2.md), [Chapter 19](ch19-operational-security.md)

!!! info "Why it matters for ToOp"

    A topology that removes every thermal overload can still be unacceptable in the control room: it may push
    fault currents above breaker ratings, or pull them below what protection needs to detect a fault. It may
    need a switching sequence that the station cannot perform, or leave a breaker to be closed across a large
    angle. ToOp models only part of this, and you need to know which part:

    - **Not modeled.** ToOp computes no short-circuit currents, checks no breaker ratings or protection
      settings, and does not plan switching sequences. Its deliverable is a set of target switch states, one
      `grid_model_id` and `open` flag per switch (`interfaces/switch_update_schema.py`,
      `dc_solver/export/export.py`), exported as a list that carries no switching order and no device type
      (`optimizer/ac/summary.py`).
    - **Proxies in the AC stage.** The voltage angle difference across open retained switches and across the
      ends of an outaged branch (`max_va_diff_n_1`, `critical_va_diff_count_n_1`, threshold
      `critical_va_diff_degree = 20.0`) approximates a synchro-check on reclosing; `voltage_jump_count_n_1`
      counts (bus, contingency) results whose voltage deviates from the base case by more than
      `critical_voltage_jump_percent = 5.0` %
      (`optimizer/interfaces/messages/ac_params.py`, `contingency/ac_loadflow_service/compute_metrics.py`).
    - **Optional cascade simulator** (pandapower only): after each contingency, lines trip on overcurrent or when
      the apparent impedance seen by a relay enters a distance-protection polygon in the R-X plane, and special
      protection schemes act on rules (`docs/contingency_analysis/cascade.md`, `contingency/pandapower/cascade/`,
      `contingency/pandapower/spps/`).
    - **Busbar outage propagation** follows protection logic: closed breakers block the outage area, closed
      disconnectors propagate it (`docs/dc_solver/busbar_outage.md`).

## Topics

- **Faults and fault calculation.** Three-phase, line-to-ground, line-to-line and double line-to-ground faults;
  symmetrical components and sequence networks (revisits [Chapter 10](../part-2-circuits/ch10-three-phase-per-unit.md));
  the Thevenin impedance at the fault bus as the diagonal entry $Z_{kk}$ of $Z_\text{bus} = Y_\text{bus}^{-1}$
  (revisits the network matrices used in [Chapter 15](ch15-ac-power-flow.md)).
- **IEC 60909 at awareness level.** Equivalent voltage source $c\,U_n/\sqrt{3}$ at the fault location; initial
  symmetrical short-circuit current $I''_k = c\,U_n / (\sqrt{3}\,|Z_k|)$ and short-circuit power
  $S''_k = \sqrt{3}\,U_n I''_k$; network feeder impedance $Z_Q = c\,U_n^2 / S''_{kQ}$; peak current
  $i_p = \kappa\sqrt{2}\,I''_k$; maximum calculations (equipment ratings) vs minimum calculations (protection
  sensitivity). SO GL Art. 30–31 require TSOs to keep short-circuit currents between these two limits at all
  times, allowing deviations only during switching sequences.
- **Topology and fault levels.** Opening a busbar coupler puts the infeeds of the two busbars in series with
  the rest of the grid instead of in parallel, so busbar splitting is a classic way to limit fault levels; closing
  couplers and meshing raise them. The same split lowers the minimum short-circuit current, which weakens
  protection sensitivity and voltage stiffness near converters.
- **Switching devices.** Circuit breakers (rated normal current, rated short-circuit breaking and making
  current), load-break switches, disconnectors (off-load only), earthing switches. Why a disconnector must not
  open load current and how that forces **make-before-break** busbar reassignment in a double-busbar station:
  couplers closed, close the target busbar disconnector, open the original one, then open the coupler if the
  station is to be split (revisits exercise 2 of [Chapter 13](ch13-substations-grid-models.md)). Moving a bay in a
  station that is already split needs temporary recoupling, which may be infeasible because of loop flows or
  fault levels. ToOp's `switching_distance` counts reassignments, not a feasible sequence.
- **Protection after a topology change.** Overcurrent pickup and grading; distance protection zones and their
  reach, infeed and outfeed effects on the apparent impedance $Z = V_\text{phase}/I$; load encroachment into
  zone 3 under heavy post-contingency loading; busbar differential protection whose zones follow the
  disconnector replica; breaker failure protection. Settings computed for one topology may not be valid for
  another.
- **ToOp's distance protection model.** The relay measures $Z = \dfrac{V_\text{LL}/\sqrt{3}}{I}$ with the angle
  of $P + jQ$ (`get_complex_impedance` in `contingency/pandapower/cascade/detection/switch_preparation.py`); the
  "danger" polygon is built from `angle`, `r_i`, `r_v`, `x_v` (`_build_poly` in
  `contingency/pandapower/cascade/detection/distance_protection.py`); the test uses $(|R|, |X|)$ divided by a
  zone factor, so the characteristic is effectively non-directional; alarm and warning zones are the same
  polygon widened by factors.
- **Closing across an angle.** Synchro-check (angle, voltage magnitude and slip limits); why reclosing a line or
  coupler across a large standing angle causes current and torque shocks on nearby machines; how the AC
  angle-difference metric approximates this check on the open switches and outaged branches of each
  contingency (`contingency/pandapower/pandapower_helpers/va_diff_info.py`, `contingency/pypowsybl/powsybl_helpers.py`).
- **Voltage, reactive power and stability at awareness level.** Splitting a station changes reactive power paths
  and voltage profiles (voltage jumps after contingencies), weakens the electrical connection of generators and
  reduces transient stability margins; cascading failures and blackouts; special protection schemes.

## Learning outcomes

- Compute three-phase fault currents with the IEC 60909 equivalent voltage source from $Z_\text{bus}$ or from
  feeder short-circuit powers, and show quantitatively how a busbar split changes them.
- Run `pandapower.shortcircuit.calc_sc` on a small grid for maximum and minimum cases, before and after opening a
  busbar coupler, and compare the results with a hand calculation.
- Draw a distance-protection zone and a load impedance point in the R-X plane and decide whether a
  post-contingency loading trips the relay, including ToOp's polygon construction.
- Write a feasible switching sequence for a busbar reassignment and list the checks (breaker ratings, protection,
  synchro-check) that a topology from ToOp still needs before operation.

## Resources

- Regulation: SO GL, Regulation (EU) 2017/1485 <https://eur-lex.europa.eu/eli/reg/2017/1485/oj/eng>:
  Art. 30–31 (short-circuit current limits and calculation), Art. 36–37 (protection and special protection
  schemes), Art. 38–39 (dynamic stability).
- Standard: IEC 60909-0:2016, *Short-circuit currents in three-phase a.c. systems, Part 0: Calculation of
  currents* <https://webstore.iec.ch/en/publication/24100> (paid; the pandapower documentation summarizes the
  method).
- Textbooks:
  - Grainger & Stevenson, *Power System Analysis* (1994): chapters "Symmetrical faults", "Symmetrical components
    and sequence networks" and "Unsymmetrical faults". (12 syllabi) **BORROW**
    [`powersystemanaly0000grai_w3w3`](https://archive.org/details/powersystemanaly0000grai_w3w3).
    ES: *Análisis de sistemas de potencia* (McGraw-Hill, 1996).
  - IEEE Press, *Protective Relaying for Power Systems II* (1992), a reprint volume of classic relaying papers.
    **BORROW** [`protectiverelayi0000unse_v7i8`](https://archive.org/details/protectiverelayi0000unse_v7i8).
  - Westinghouse, *Electrical Transmission and Distribution Reference Book* (1964): symmetrical components, fault
    calculation and relaying chapters. **BORROW**
    [`electricaltransm0000cent`](https://archive.org/details/electricaltransm0000cent).
  - Anderson, *Analysis of Faulted Power Systems* (1973), the reference for unsymmetrical faults.
    **PRINT-DISABLED** [`analysisoffaulte0000ande`](https://archive.org/details/analysisoffaulte0000ande).
  - Horowitz & Phadke, *Power System Relaying* (2nd ed. 1995). **PRINT-DISABLED**
    [`powersystemrelay0000horo_ed02`](https://archive.org/details/powersystemrelay0000horo_ed02).
  - Machowski, Bialek & Bumby, *Power System Dynamics and Stability* (1st ed. 1997; later editions as *Power
    System Dynamics: Stability and Control*), for switching, reclosing and stability effects.
    **PRINT-DISABLED** [`powersystemdynam0000mach`](https://archive.org/details/powersystemdynam0000mach).
  - Schlabbach, *Short-Circuit Currents* (IET, 2005), a practical IEC 60909 text. **NOT ON IA**.
  - Blackburn, *Protective Relaying: Principles and Applications*: **NOT ON IA**.
- Free: Barabási, *Network Science*, Ch. 8 (robustness and cascading failures), **FREE**
  <https://networksciencebook.com/chapter/8>.
- Documentation:
  - pandapower short-circuit calculation <https://pandapower.readthedocs.io/en/latest/shortcircuit.html> and
    `calc_sc` <https://pandapower.readthedocs.io/en/latest/shortcircuit/run.html>; pandapower protection module
    (fuses, overcurrent relays) <https://pandapower.readthedocs.io/en/latest/protection.html>.
  - ToOp: `docs/contingency_analysis/cascade.md`, `docs/contingency_analysis/propagation.md`,
    `docs/dc_solver/busbar_outage.md`.

## Worked example

Two 220 kV infeeds reach a double-busbar station: infeed 1 with $S''_{k1} = 8\,000$ MVA through a 40 km line,
infeed 2 with $S''_{k2} = 6\,000$ MVA through a 60 km line, both lines $j0.4\ \Omega/\text{km}$. Take $c = 1.1$
and neglect resistance and other paths.

$$
Z_{Q1} = \frac{1.1 \cdot 220^2}{8\,000} = 6.66\ \Omega, \qquad
Z_{Q2} = \frac{1.1 \cdot 220^2}{6\,000} = 8.87\ \Omega .
$$

Path impedances are $Z_1 = 6.66 + 16 = 22.66\ \Omega$ and $Z_2 = 8.87 + 24 = 32.87\ \Omega$.

- Coupler closed: $Z_k = Z_1 \parallel Z_2 = 13.41\ \Omega$, so
  $I''_k = 1.1 \cdot 220 / (\sqrt{3} \cdot 13.41) = 10.4$ kA on both busbars.
- Coupler open: busbar 1 sees $Z_1$ only, $I''_k = 6.2$ kA; busbar 2 sees $Z_2$ only, $I''_k = 4.3$ kA.

If the bay breakers are rated 8 kA breaking current, the split is what makes the station operable; if the busbar
2 protection needs at least 5 kA to operate selectively, the same split is not acceptable. A flow-only optimizer
sees neither constraint.

## Documentation vs code: known discrepancies

- The relay table example in `docs/contingency_analysis/cascade.md` (`angle` 80°, `r_i` 2.5 Ω, `r_v` 10 Ω,
  `x_v` 15 Ω) produces a **self-intersecting** polygon with `_build_poly`, because
  $r_v \tan(\text{angle}) = 56.7\ \Omega > x_v$; shapely reports the geometry as invalid. The unit tests use 30°.
- The `changing_switches_to_orao_dict` docstring (`optimizer/ac/summary.py`) shows PST tap and
  `TERMINALS_CONNECTION` actions, but the function emits only `SWITCH` entries, and the switch updates carry no
  ordering or device type, so they are not a switching sequence.

## Lab

!!! example "Lab: fault levels before and after a busbar split, and relay zones in the R-X plane"

    Tools: Jupyter with pandapower, NumPy and matplotlib ([Chapter 30](../part-5-computing/ch30-scientific-python.md));
    GeoGebra (<https://www.geogebra.org/calculator>) or matplotlib for the R-X plane.

    **Part A: short circuit with pandapower.**

    1. Build the worked example grid at 220 kV: external grids at buses G1 and G2 (`s_sc_max_mva`,
       `s_sc_min_mva`, `rx_max`, `rx_min` set), station busbars S1 and S2 joined by a bus-bus switch of type
       `"CB"`, lines G1–S1 (40 km), G2–S2 (60 km), S1–L and S2–L (30 km) and G1–G2 (150 km), and a load at L.
    2. Run `pandapower.shortcircuit.calc_sc(net, case="max", ip=True)` with the coupler closed and open, and
       tabulate `net.res_bus_sc` (`ikss_ka`, `skss_mw`, `ip_ka`). Keep `branch_results=False`: in pandapower
       3.5.4 branch results fail on a grid with a bus-bus switch.
    3. Repeat with `case="min"`. Explain the extra paths (via L and G1–G2) that make the pandapower results
       differ from the hand calculation, then remove them and match the hand calculation.
    4. Run `pandapower.runpp` in both states and record the voltage angle difference between S1 and S2 with the
       coupler open: the angle a synchro-check would see when recoupling.
    5. Choose a breaker rating and a minimum fault current for protection and state, for each busbar, whether
       the split is acceptable.

    **Part B: distance zones in the R-X plane.**

    1. In GeoGebra or matplotlib, draw ToOp's polygon with vertices $(0,0)$, $(r_i,0)$, $(r_i, r_i\tan\theta)$,
       $(r_v, r_v\tan\theta)$, $(r_v, x_v)$, $(0, x_v)$ for $\theta = 30°$, $r_i = 2.5$, $r_v = 10$, $x_v = 15$ Ω,
       with sliders for the four parameters.
    2. Plot the apparent impedance of a 380 kV line carrying $S = P + jQ$:
       $|Z| = V_\text{LL}^2/|S|$ at the angle of $P + jQ$. Move $S$ along a post-contingency loading path and
       find the MVA at which the point enters the zone.
    3. Apply a warning factor of 1.41 as ToOp does (divide $|R|$ and $|X|$ by the factor) and find the new
       tripping MVA. Show the effect of taking absolute values for a reverse power flow.
    4. Move $\theta$ to 80° and observe the self-intersection noted above; derive the validity condition
       $r_v \tan\theta \le x_v$.

## Exercises

1. Pen and paper: using symmetrical components, derive the single line-to-ground fault current
   $I_f = 3E/(Z_1 + Z_2 + Z_0)$ and explain why solidly grounded transmission grids can have higher single-phase
   than three-phase fault currents near transformers with low zero-sequence impedance.
2. *(Jupyter: NumPy)* For a 5-bus grid given by its $Y_\text{bus}$, compute $Z_\text{bus}$ and every bus's
   $I''_k$. Then model a busbar split as in [Chapter 18](ch18-sensitivity-factors-2.md) (add a node, move two
   branches) and recompute. Which buses changed by more than 10 %, and why do remote buses barely change?
3. *(Jupyter: pandapower)* On `pandapower.networks.case9()`, add the short-circuit data `calc_sc` needs
   (external grid `s_sc_max_mva`, `s_sc_min_mva`, `rx_max`, `rx_min`; generator `vn_kv`, `sn_mva`, `xdss_pu`,
   `rdss_ohm`, `cos_phi`; line `endtemp_degree` for the minimum case), compare `case="max"` and `case="min"` for
   all buses and explain the role of $c_\text{max}$ and $c_\text{min}$ and of the line end temperature.
4. Pen and paper: write the full switching sequence to move one line bay from busbar 1 to busbar 2 in a
   double-busbar, single-breaker station and then split the station, marking which devices are breakers and
   which are disconnectors, and the moments at which the line is in parallel across both busbars.
5. *(GeoGebra)* Draw a quadrilateral zone 1 (80 % of the line impedance) and zone 2 (120 %) for a 100 km line
   with $z = 0.03 + j0.3\ \Omega/\text{km}$. Add an infeed from a busbar that ToOp split so that half the fault
   current comes from the remote end; show how the apparent impedance for a fault at 90 % of the line moves
   and whether zone 1 under- or over-reaches.
6. *(Jupyter: pandas)* In the AC results of the [Chapter 19](ch19-operational-security.md) lab, list the
   `va_diff_results` rows above 20° and 10°. For the worst one, name the switch or outaged branch and describe
   what an operator would have to do before reclosing it.
