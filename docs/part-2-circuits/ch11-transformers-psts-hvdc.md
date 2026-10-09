# Chapter 11. Transformers, phase-shifting transformers, machines and HVDC

**Weeks:** 3 · **Semester:** 2 · **Prerequisites:** [Chapter 10](ch10-three-phase-per-unit.md)

!!! info "Why it matters for ToOp"

    PST tap optimization is one of ToOp's four action types. The solver models PSTs through the phase-shift
    distribution factor (`dc_solver/preprocess/helpers/psdf.py`) and writes the angle of the selected tap, read
    from a per-tap table, into the nodal injection vector (`dc_solver/jax/pst.py`); per-tap susceptances cover
    PSTs whose reactance changes with the tap. The PowSyBl importer flags a PST as linear in DC when its
    reactance is the same at every tap (`get_linear_pst` in `dc_solver/preprocess/powsybl/powsybl_helpers.py`).
    How tap tables enter the sensitivity updates is taught in
    [Chapter 18](../part-3-power-systems/ch18-sensitivity-factors-2.md). Transformer ratios enter the DC
    reactance as $x/\rho$ (`get_trafos` in the same file). In the pandapower backend, three-winding transformers
    are represented around an auxiliary star bus and their outage is a multi-outage (`get_trafo3w_multioutage`
    in `dc_solver/preprocess/pandapower/pandapower_backend.py`); the PowSyBl backend asserts that there are
    none (`dc_solver/preprocess/powsybl/powsybl_backend.py`). HVDC converter stations (LCC and VSC) and
    batteries are fixed injections (`_get_injections` in `powsybl_backend.py`).

    **Awareness:** HVDC setpoints exist in the action-set schema (`HVDCRange` with `min_power` and
    `max_power` in `interfaces/stored_action_set.py`), but the field is documented as "currently not
    implemented yet in the solver", and preprocessing always writes `hvdc_ranges=[]`
    (`dc_solver/preprocess/network_data.py`). Other topology optimization tools may optimize them.

## Topics

- Magnetic circuits; ideal and real transformers; equivalent circuits; T vs π models.
- Tap changers and off-nominal turns ratios; **phase-shifting transformers** (symmetric and asymmetric; tap
  tables; linear vs non-linear steps). The phase shift is a multiplication by $e^{j\alpha}$
  (revisits [Chapter 1](../part-1-math/ch01-complex-numbers.md)); ToOp's tap tables are detailed in
  [Chapter 18](../part-3-power-systems/ch18-sensitivity-factors-2.md).
- Three-winding transformers and the star equivalent.
- Synchronous generator at awareness level: voltage and reactive power control (PV behavior).
- Power electronics, HVDC (LCC and VSC), SVCs and batteries at awareness level; HVDC links as controllable
  pairs of injections.

## Learning outcomes

- Refer impedances across an ideal transformer and build the star equivalent of a three-winding transformer.
- Draw the phasor diagram of symmetric and quadrature PSTs and compute the phase shift and magnitude change
  of a tap step.
- Show with the DC model that the flow through a PST branch in a meshed network is affine in the phase shift
  $\alpha$, and compute its slope.

## Resources

- Courses and notes: MIT OCW 6.061 notes Ch. 6 (magnetic circuits) and Ch. 9 (synchronous machines); MIT
  OCW 6.685 *Electric Machines* <https://ocw.mit.edu/courses/6-685-electric-machines-fall-2013/>; MIT OCW
  6.622 *Power Electronics* (awareness) <https://ocw.mit.edu/courses/6-622-power-electronics-spring-2023/>.
- Free texts: Kuphaldt Vol. II, Ch. 9 (transformers); PowSyBl transformer model documentation (cited in ToOp;
  the original page `www.powsybl.org/pages/documentation/grid/model/` is gone, archived copy of August 2024:
  <https://web.archive.org/web/20240816020722/https://www.powsybl.org/pages/documentation/grid/model/#transformers>).
- Textbooks:
  - Fitzgerald, Kingsley & Umans, *Electric Machinery* (6 syllabi): Ch. 1 magnetic circuits, Ch. 2
    transformers (with per unit), Ch. 5 synchronous machines (6th ed. numbering). 6th ed. **BORROW**
    [`isbn_9780070530393`](https://archive.org/details/isbn_9780070530393).
    ES: *Máquinas eléctricas* (McGraw-Hill; 6th ed. required at UPC and U. de Chile).
  - Chapman, *Electric Machinery Fundamentals* (5 syllabi): Ch. 1–2 and 4 (5th ed.). 5th ed. international
    **BORROW** [`electricmachiner0000chap_w2e6`](https://archive.org/details/electricmachiner0000chap_w2e6).
    ES: *Máquinas eléctricas* (McGraw-Hill; 5th ed. required at UPC).
  - MIT EE Staff, *Magnetic Circuits and Transformers* (2 syllabi, UTN and FIUBA), a classic. **BORROW**
    [`magneticcircuits0000eest`](https://archive.org/details/magneticcircuits0000eest).
    ES: *Circuitos magnéticos y transformadores* (Reverté).
  - Hughes, *Electrical and Electronic Technology*, 10th ed., one volume from circuits to machines.
    **BORROW** [`hugheselectrical0000hugh_k4v8`](https://archive.org/details/hugheselectrical0000hugh_k4v8).
  - Fraile Mora, *Máquinas eléctricas* (Spanish original; 2 syllabi, UPC and FIUBA). **NOT ON IA**;
    6th ed. **USER-UPLOAD (ES)** [`maquinas-electricas-6a.-ed.-fraile-mora-jesus.`](https://archive.org/details/maquinas-electricas-6a.-ed.-fraile-mora-jesus.).
- Documentation and tools: GeoGebra Classic <https://www.geogebra.org/classic>; NumPy `numpy.linalg`
  <https://numpy.org/doc/stable/reference/routines.linalg.html>.

## Worked example

Buses 1 and 2 are joined by two parallel paths: line $a$ (reactance $x_a$) in series with an ideal PST that
advances the voltage on its side by $\alpha$, and line $b$ (reactance $x_b$). Bus 1 injects $P$ and bus 2 withdraws
it. In the DC model, with $\theta_{12} = \theta_1 - \theta_2$ and $\alpha$ in radians,

$$
F_a = \frac{\theta_{12} + \alpha}{x_a},
\qquad
F_b = \frac{\theta_{12}}{x_b},
\qquad
F_a + F_b = P
\quad\Rightarrow\quad
F_a = \frac{x_b}{x_a + x_b}\,P + \frac{\alpha}{x_a + x_b}.
$$

The flow is affine in $\alpha$: a base share of $P$ plus a loop flow $\alpha/(x_a + x_b)$ that circulates through
both lines. With $x_a = 0.1$ pu, $x_b = 0.3$ pu and $P = 1$ pu, $F_a = 0.75$ pu at $\alpha = 0$ and each degree
adds $(\pi/180)/0.4 \approx 0.0436$ pu, 4.36 MW on a 100 MVA base; this is why ToOp scales its PSDF by
`base_mva * (np.pi / 180)`. With a single line and no parallel path, the PST could not change the flow at all.
The sign of $\alpha$ here is a modeling choice; ToOp's sign conventions are covered in Chapters
[17](../part-3-power-systems/ch17-sensitivity-factors-1.md) and
[18](../part-3-power-systems/ch18-sensitivity-factors-2.md).

## Lab

!!! example "Lab: phase shifters as phasors and as loop flows"

    Tools: GeoGebra Classic, then Jupyter + NumPy + matplotlib (reusing `nodal_solve` from the
    [Chapter 8](ch08-dc-circuits.md) lab).

    1. In GeoGebra, let `U = 1 + 0i` be the input voltage and `k` a slider in $[-0.2, 0.2]$. A quadrature booster
       adds a voltage perpendicular to the input: `W_q = U + k i U`. A symmetric PST rotates the input:
       `W_s = U (cos(α) + i sin(α))` with `α = arg(W_q)`. Compare `abs(W_q)` and `abs(W_s)` and explain why the
       symmetric design keeps the magnitude.
    2. Add a slider `φ` for the step angle and the general series voltage `W_g = U + k (cos(φ) + i sin(φ))`.
       Check `arg(W_g)` against the formula of [Chapter 1](../part-1-math/ch01-complex-numbers.md), Exercise 4.
    3. In a notebook, compute $F_a$ and $F_b$ of the worked example for $\alpha$ from $-10^\circ$ to $10^\circ$, plot
       them and check the slope. At which $\alpha$ does line $b$ carry no flow, and when does its flow reverse?
    4. Model the PST as an equivalent pair of injections: $-\alpha/x_a$ at bus 1 and $+\alpha/x_a$ at bus 2, plus
       $\alpha/x_a$ added back to line $a$'s flow. Verify with `nodal_solve` that this gives the same flows as
       the direct formula.
    5. Put a PST on branch `(1, 3)` of the [Chapter 2](../part-1-math/ch02-linear-algebra.md) network (with
       susceptances instead of conductances) and compute the change of every branch flow per degree by
       finite differences. This column of sensitivities is what ToOp's PSDF stores.

## Exercises

1. An ideal 380/110 kV transformer feeds a 10 Ω impedance on its 110 kV side. What impedance is seen from the
   380 kV side? *(Pen and paper)* *Answer:* $10 \cdot (380/110)^2 \approx 119.3\ \Omega$.
2. ToOp replaces a transformer's reactance $x$ by $x/\rho$ in the DC model (`get_trafos` in
   `dc_solver/preprocess/powsybl/powsybl_helpers.py`, commented as equivalent to PowSyBl's
   `dc_use_transformer_ratio = True`). For $\rho = 1.05$, by what percentage does the branch susceptance change,
   and does the transformer attract more or less flow in a meshed network? *(Pen and paper)*
3. A three-winding transformer has pairwise short-circuit reactances $x_{12} = 0.10$, $x_{13} = 0.25$ and
   $x_{23} = 0.20$ pu on a common base. Compute the star equivalent $x_1 = (x_{12} + x_{13} - x_{23})/2$ and its
   cyclic permutations, and explain why an outage of this transformer removes three branches at once.
   *(Pen and paper)* *Answer:* $x_1 = 0.075$, $x_2 = 0.025$, $x_3 = 0.175$ pu.
4. Build two tap tables with pandas for a PST with taps $-16 \dots 16$: one with constant $x$ and $\alpha$ from
   the atan2 formula of [Chapter 1](../part-1-math/ch01-complex-numbers.md) (1% per step at $90^\circ$), one where
   $x$ grows by 1% per tap away from neutral. Apply the same test as `get_linear_pst` in DC mode (`np.allclose`
   on `x`) to each. Is $\alpha$ exactly proportional to the tap number in the first table? *(Jupyter: pandas +
   NumPy)*
5. Draw the phasor diagram of a symmetric PST and a quadrature booster with the same phase shift in
   GeoGebra, and measure the output magnitude of each. *(GeoGebra)*
6. An HVDC link between buses 1 and 3 of a three-bus AC ring is a pair of injections $-P_{dc}$ at bus 1 and
   $+P_{dc}$ at bus 3. With `nodal_solve`, show that every AC branch flow is affine in $P_{dc}$ and find the
   setpoint that unloads the most loaded AC line. Why would optimizing $P_{dc}$ be a continuous action,
   unlike a busbar split? *(Jupyter: NumPy + matplotlib)*
