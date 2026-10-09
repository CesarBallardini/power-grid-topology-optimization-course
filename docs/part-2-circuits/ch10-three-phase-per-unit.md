# Chapter 10. Three-phase systems and the per-unit system

**Weeks:** 1.5 · **Semester:** 2 · **Prerequisites:** [Chapter 9](ch09-ac-steady-state.md)

!!! info "Why it matters for ToOp"

    Limits are converted between current and power with $P = \sqrt{3}\, V_{LL} I$: the PowSyBl preprocessing
    computes `p_limit` as current limit × limit voltage × `1e-3` × `sqrt(3)`
    (`dc_solver/preprocess/powsybl/powsybl_helpers.py`). The cascade relay model uses phase voltages
    $V_{LL}/\sqrt{3}$ to compute impedances (`contingency/pandapower/cascade/detection/switch_preparation.py`).
    Network data is converted to per unit, and the phase-shift distribution factors are scaled by the base
    power, `base_mva * (np.pi / 180) * psdf` (`dc_solver/preprocess/helpers/psdf.py`, with the base from
    `get_base_mva` in `dc_solver/preprocess/powsybl/powsybl_backend.py`). ToOp's AC metrics also report
    voltage-angle differences (`max_va_diff_n_0`, `docs/topology_optimizer/metrics.md`); why the angle across an
    open breaker matters for closing it (synchro-check) is left to
    [Chapter 20](../part-3-power-systems/ch20-protection-short-circuit.md).

## Topics

- Balanced three-phase sources and loads; wye and delta; line vs phase quantities; three-phase power
  $S_{3\phi} = \sqrt{3}\, V_{LL} I_L$, $P = \sqrt{3}\, V_{LL} I_L \cos\varphi$.
- The operator $a = e^{j2\pi/3}$ (revisits the roots of unity of
  [Chapter 1](../part-1-math/ch01-complex-numbers.md)); per-phase equivalent circuit.
- Per-unit system: base quantities, change of base, per-unit transformer models.
- Symmetrical components (awareness level; used for faults, not by ToOp).

## Learning outcomes

- Draw the phasor diagram of a balanced three-phase set and derive $|V_{LL}| = \sqrt{3}\,|V_{\text{ph}}|$ and the
  $30^\circ$ shift.
- Convert current limits to power limits and back at a given voltage and power factor.
- Convert impedances to per unit and change the base of a per-unit value.

## Resources

- Free texts: Kuphaldt Vol. II, Ch. 10 "Polyphase AC Circuits"; MIT 6.061 notes Ch. 3 (polyphase networks)
  and Ch. 4 (symmetrical components); Overbye, ECEN 615 Lecture 3 (per unit, Ybus)
  <https://overbye.engr.tamu.edu/wp-content/uploads/sites/146/2022/09/Lecture-3.pdf>.
- Textbooks: the three-phase chapter of any [Chapter 8](ch08-dc-circuits.md) circuit book; Chapman ([Chapter 11](ch11-transformers-psts-hvdc.md)) Ch. 2 for per-unit
  transformer models.
- Documentation and tools: GeoGebra Classic <https://www.geogebra.org/classic>; pandas
  <https://pandas.pydata.org/docs/>; GNU Octave "Linear Algebra" <https://docs.octave.org/latest/Linear-Algebra.html>.

## Worked example

With a three-phase base power $S_{\text{base}}$ and a line-to-line base voltage $V_{\text{base}}$:

$$
Z_{\text{base}} = \frac{V_{\text{base}}^2}{S_{\text{base}}},
\qquad
I_{\text{base}} = \frac{S_{\text{base}}}{\sqrt{3}\, V_{\text{base}}},
\qquad
z_{\text{pu}} = \frac{Z}{Z_{\text{base}}},
\qquad
z_{\text{pu,new}} = z_{\text{pu,old}} \left(\frac{V_{\text{base,old}}}{V_{\text{base,new}}}\right)^2
\frac{S_{\text{base,new}}}{S_{\text{base,old}}}.
$$

- A 380 kV line rated 1 000 A carries at most $P = \sqrt{3} \cdot 380\ \text{kV} \cdot 1000\ \text{A} \approx 658$ MW at
  unity power factor.
- A 220 kV line of 100 km with $x = 0.4\ \Omega/\text{km}$ has $X = 40\ \Omega$. On a 100 MVA base,
  $Z_{\text{base}} = 220^2/100 = 484\ \Omega$, so $x_{\text{pu}} = 40/484 \approx 0.0826$, and
  $I_{\text{base}} \approx 262$ A.
- A transformer with $x = 0.10$ pu on its own 50 MVA rating has $x = 0.10 \cdot 100/50 = 0.20$ pu on a 100 MVA
  system base (same voltage base).

## Lab

!!! example "Lab: three-phase phasors and a per-unit branch table"

    Tools: GeoGebra Classic, then Jupyter + pandas + NumPy.

    1. In GeoGebra, define `V_a = 1 + 0i`, `V_b = V_a (cos(-120°) + i sin(-120°))` and
       `V_c = V_a (cos(120°) + i sin(120°))`, and draw them with `Vector(V_a)` and so on. Add
       `V_ab = V_a - V_b` and check `abs(V_ab)` and `arg(V_ab)`: $\sqrt{3}$ and $30^\circ$.
    2. Add a slider `φ` and balanced currents `I_a = V_a (cos(-φ) + i sin(-φ))`, and likewise for `I_b` and
       `I_c`. Check that `I_a + I_b + I_c = 0` for every `φ` (no neutral current), and read the power factor
       angle between each voltage and its current.
    3. In a notebook, build a pandas `DataFrame` of branches with columns `name`, `v_kv`, `length_km`,
       `r_ohm_per_km`, `x_ohm_per_km`, `i_max_a`, for example a 380 kV line of 150 km (0.03, 0.30 Ω/km,
       2 000 A), a 220 kV line of 100 km (0.06, 0.40 Ω/km, 1 000 A) and a 110 kV cable of 20 km (0.05, 0.12 Ω/km,
       600 A). Add columns for $Z_{\text{base}}$, $r_{\text{pu}}$, $x_{\text{pu}}$ on a 100 MVA base and the
       limit in MW at unity power factor.
    4. Compare your MW column with the formula used in ToOp's `powsybl_helpers.py` (`value * limit_voltage *
       1e-3 * math.sqrt(3)`) and state the units each factor assumes.

## Exercises

1. A balanced wye load has $Z_Y = 30 + j40\ \Omega$ per phase on a 400 V (line-to-line) supply. Compute the line
   current, total $P$ and $Q$, and the equivalent delta impedance $Z_\Delta = 3 Z_Y$. *(Pen and paper)*
2. Verify numerically that the instantaneous total power of a balanced three-phase set is constant in time,
   unlike single-phase power. *(Jupyter: NumPy + matplotlib)*
3. The cascade relay model computes a phase voltage $V_{LL}/\sqrt{3}$, an impedance magnitude $|Z| = V_{\text{ph}}/I$
   and an angle $\operatorname{atan2}(Q, P)$ (`get_complex_impedance` in
   `contingency/pandapower/cascade/detection/switch_preparation.py`). Compute $R$ and $X$ seen by a relay at
   380 kV with $I = 1\,000$ A, $P = 600$ MW and $Q = 200$ Mvar. *(Pen and paper)* *Answer:* $|Z| \approx 219.4\ \Omega$,
   $R \approx 208.1\ \Omega$, $X \approx 69.4\ \Omega$.
4. A 1 600 MVA transformer has $x = 0.15$ pu on its own rating. Express it on a 100 MVA base, and explain why
   the per-unit value does not depend on which side's voltage base is used when bases are chosen through the
   turns ratio. *(Pen and paper)*
5. Symmetrical components (awareness): with $A = \begin{bmatrix} 1 & 1 & 1 \\ 1 & a^2 & a \\ 1 & a & a^2 \end{bmatrix}$,
   compute $V_{012} = A^{-1} V_{abc}$ for a set where phase b has dropped to 0.9 pu. *(Octave)* *Answer:*
   $|V_1| \approx 0.967$, $|V_0| = |V_2| \approx 0.033$ pu.
