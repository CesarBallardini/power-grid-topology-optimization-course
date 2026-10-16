# Chapter 15. AC power flow

**Weeks:** 2.5 · **Semester:** 3 · **Prerequisites:** [Chapter 6](../../part-1-math/ch06-newton-raphson/index.md), [Chapter 14](../ch14-component-models/index.md)

!!! info "Why it matters for ToOp"

    The final word on every candidate topology is an AC N-1 analysis
    (`contingency/ac_loadflow_service/`, pandapower and pypowsybl runners). ToOp's AC-only metrics (voltage jump
    after a contingency, voltage-angle difference across open breakers, non-converging load flows) come from
    here, and so does the AC–DC mismatch correction applied to DC flows
    ([Chapter 17](../ch17-sensitivity-factors-1/index.md)).

    Two AC settings shape everything downstream:

    - **Distributed slack.** ToOp's PowSyBl parameter sets (`grid_helpers/powsybl/loadflow_parameters.py`) use
      OpenLoadFlow with `distributed_slack=True`: `POWSYBL_LOADFLOW_PARAM_PF` balances with
      `BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX`, while `CGMES_DISTRIBUTED_SLACK` and
      `UCTE_DISTRIBUTED_SLACK` use `BalanceType.PROPORTIONAL_TO_GENERATION_P` with
      `slackDistributionFailureBehavior = LEAVE_ON_SLACK_BUS`. `docs/dc_solver/loadflow_parameters.md` justifies
      the distributed slack by the smaller AC–DC mismatch it produces. The DC solver itself, in contrast, uses a
      single slack bus ([Chapter 17](../ch17-sensitivity-factors-1/index.md)).
    - **Initialization and fallbacks.** On import, `find_converging_loadflow_params`
      (`importer/pypowsybl_import/preprocessing.py`) tries `POWSYBL_LOADFLOW_PARAM_PF` and then
      `CGMES_DISTRIBUTED_SLACK`, each with the voltage initializations `PREVIOUS_VALUES`, `DC_VALUES` and
      `UNIFORM_VALUES`, and keeps the first combination whose AC load flow converges. If none converges it
      raises an error, unless `fail_on_non_convergence` is off, in which case it logs a warning and continues
      with `CGMES_DISTRIBUTED_SLACK`. The PowSyBl DC backend (`dc_solver/preprocess/powsybl/powsybl_backend.py`)
      runs its own AC load flow (default parameters `CGMES_DISTRIBUTED_SLACK`); if it does not converge, the
      AC–DC mismatch is set to zero.

## Topics

- Power balance equations in polar and rectangular form; bus types (PQ, PV, slack).
- Newton–Raphson power flow: mismatch vector, Jacobian blocks, iteration (revisits
  [Chapter 6](../../part-1-math/ch06-newton-raphson/index.md) on a real system).
- Gauss–Seidel; fast decoupled load flow (the $B'$ and $B''$ matrices, MATPOWER `makeB`).
- Reactive power limits and PV→PQ switching; voltage control.
- Distributed slack: participation factors and how they change the Jacobian; OpenLoadFlow balance types.
- Initialization (flat, DC, previous values); convergence failures and what they indicate (voltage collapse,
  islands, bad data); why the initialization can decide which solution you get.
- Tools: MATPOWER `runpf` and `mpoption` in Octave; `pandapower.runpp`; `pypowsybl.loadflow.run_ac`;
  optionally PowerWorld Simulator, the commercial tool used in the labs of several surveyed courses
  ([Appendix A](../../appendices/a-syllabi.md)).

## Learning outcomes

- Implement a Newton–Raphson power flow for a 5-bus system and match pandapower's (or MATPOWER's) results.
- Write the four Jacobian blocks in polar form and check them against finite differences.
- Explain why an AC power flow can fail to converge for a topology that the DC model accepts.
- Compare single-slack and distributed-slack solutions of the same case and explain the difference in
  generator outputs and branch flows.

## Resources

- Courses and videos:
  - NPTEL *Computer Aided Power System Analysis* (IIT Roorkee, Biswarup Das)
    <https://nptel.ac.in/courses/108107127>: Lectures 6 (power flow equations, bus types), 7–10 (Gauss–Seidel),
    11–15 (Newton–Raphson in polar coordinates: Jacobian, algorithm, example), 16–19 (rectangular
    coordinates, generator reactive power limits) and 20–22 (fast decoupled load flow).
  - NPTEL *Power System Analysis* (IIT Kharagpur, A. K. Sinha) <https://nptel.ac.in/courses/108105067>:
    Lectures 16–22 (Power Flow I–VII) and 24 (review of power flow study).
  - Overbye, TAMU ECEN 615 *Methods of Electric Power System Analysis*, Fall 2025 (slides, no videos)
    <https://overbye.engr.tamu.edu/ecen-615-fall-2025/lecture-notes/>: Lectures 4–5 (Newton–Raphson, decoupled
    and fast decoupled power flow) and 7–8 (advanced power flow: tap changers, phase shifters, reactive limits).
- Free: MATPOWER manual (§4.1 AC power flow, Table 4-2 power flow options)
  <https://matpower.org/docs/MATPOWER-manual.pdf>; pandapower power flow tutorials
  <https://github.com/e2nIEE/pandapower/tree/master/tutorials> and AC power flow documentation
  <https://pandapower.readthedocs.io/en/latest/powerflow/ac.html>; PowSyBl OpenLoadFlow parameters
  <https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/loadflow/parameters.html>.
- Free: PowerWorld training module I7 "Power Flow Analysis and Voltage Control" (slides)
  <https://www.powerworld.com/training/courses/introduction-to-powerworld-simulator/voltage-control>, background
  for lab step 6. The same channel has the video (vendor training, free) "Training: Power Flow Analysis and
  Voltage Control" <https://www.youtube.com/watch?v=Id-PWarUXGc>, and from the Training playlist
  <https://www.youtube.com/playlist?list=PL0B37D8DFCE4A53A4>: "S1: System Modeling Methods and Equations"
  <https://www.youtube.com/watch?v=YGeNNh0FFhg>, "S2: Advanced Power Flow Options"
  <https://www.youtube.com/watch?v=D5qcmmSTem8>, "S3: Techniques for Conditioning Hard-to-Solve Cases" (see the
  convergence failures of lab step 3) <https://www.youtube.com/watch?v=wA1mIhvCQvY>, "S6: PV Curves"
  <https://www.youtube.com/watch?v=0tYvU8KInHo>, and "S7: QV Curves" <https://www.youtube.com/watch?v=pX1wNcTFpYQ>
  (voltage collapse).
- Software: GNU Octave <https://octave.org/> with MATPOWER <https://matpower.org/>; GeoGebra
  <https://www.geogebra.org/classic>; PowerWorld Simulator demo (Windows only, up to 13 buses)
  <https://www.powerworld.com/download-purchase/demo-software>.
- Papers: Tinney & Hart, "Power flow solution by Newton's method", *IEEE Trans. PAS* 86(11), 1967; Stott &
  Alsaç, "Fast decoupled load flow", *IEEE Trans. PAS* 93(3), 1974.
- Textbooks:
  - Wood, Wollenberg & Sheblé, *Power Generation, Operation, and Control*, 3rd ed. (2014), Ch. 6: §6.8
    Newton–Raphson, §6.16 decoupled power flow, §6.17 Gauss–Seidel (verified against the publisher's table
    of contents). 1st and 2nd eds. **PRINT-DISABLED**; 3rd ed. **NOT ON IA**.
  - Power flow chapters of Grainger & Stevenson, Glover et al., Saadat ([Chapter 14](../ch14-component-models/index.md)).
- ToOp: `docs/dc_solver/loadflow_parameters.md`, `grid_helpers/powsybl/loadflow_parameters.py`.

## Worked example

**Power flow equations.** With $V_i = |V_i| e^{j\theta_i}$, $Y_\text{bus} = G + jB$ and
$\theta_{ik} = \theta_i - \theta_k$, the complex injections are $S = [V]\,\overline{Y_\text{bus} V}$, or per bus

$$
P_i = \sum_{k} |V_i||V_k|\left(G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}\right), \qquad
Q_i = \sum_{k} |V_i||V_k|\left(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}\right).
$$

**Newton–Raphson.** The unknowns are $\theta$ at all non-slack buses and $|V|$ at PQ buses. Each iteration solves

$$
\begin{bmatrix} J_{11} & J_{12} \\ J_{21} & J_{22} \end{bmatrix}
\begin{bmatrix} \Delta\theta \\ \Delta|V| \end{bmatrix}
= \begin{bmatrix} P^\text{spec} - P(\theta, |V|) \\ Q^\text{spec} - Q(\theta, |V|) \end{bmatrix},
\qquad
J_{11} = \frac{\partial P}{\partial \theta},\;
J_{12} = \frac{\partial P}{\partial |V|},\;
J_{21} = \frac{\partial Q}{\partial \theta},\;
J_{22} = \frac{\partial Q}{\partial |V|},
$$

keeping the rows of $P$ for non-slack buses and of $Q$ for PQ buses. For $k \neq i$:

$$
\frac{\partial P_i}{\partial \theta_k} = |V_i||V_k|(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}),\quad
\frac{\partial P_i}{\partial |V_k|} = |V_i|(G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}),
$$

$$
\frac{\partial Q_i}{\partial \theta_k} = -|V_i||V_k|(G_{ik}\cos\theta_{ik} + B_{ik}\sin\theta_{ik}),\quad
\frac{\partial Q_i}{\partial |V_k|} = |V_i|(G_{ik}\sin\theta_{ik} - B_{ik}\cos\theta_{ik}),
$$

and on the diagonal

$$
\frac{\partial P_i}{\partial \theta_i} = -Q_i - B_{ii}|V_i|^2,\quad
\frac{\partial P_i}{\partial |V_i|} = \frac{P_i}{|V_i|} + G_{ii}|V_i|,\quad
\frac{\partial Q_i}{\partial \theta_i} = P_i - G_{ii}|V_i|^2,\quad
\frac{\partial Q_i}{\partial |V_i|} = \frac{Q_i}{|V_i|} - B_{ii}|V_i|.
$$

The same blocks in matrix form, as in the `dSbus_dV` function of MATPOWER and of its PYPOWER port in pandapower
(`pandapower.pypower.dSbus_dV`), with $I = Y_\text{bus} V$ and
$E = [V/|V|]$ (a diagonal matrix):

$$
\frac{\partial S}{\partial \theta} = j[V]\,\overline{\left([I] - Y_\text{bus}[V]\right)}, \qquad
\frac{\partial S}{\partial |V|} = [V]\,\overline{Y_\text{bus} E} + \overline{[I]}\,E,
$$

with $J_{11}, J_{21}$ the real and imaginary parts of $\partial S/\partial\theta$ and $J_{12}, J_{22}$ those of
$\partial S/\partial|V|$. Check your implementation against finite differences before trusting it:

```python title="ac_jacobian.py"
--8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/ac_jacobian.py"
```

`test_ac_jacobian.py`, next to it in the chapter's `src` folder, does exactly that: central differences of
`injections` on a random three-bus state, column by column.

**Distributed slack.** Replace the slack bus's unknown $P$ by a scalar mismatch $\Delta$ shared by the
participating generators, $P_{g,i} = P_{g,i}^0 + k_i \Delta$ with $\sum_i k_i = 1$. The unknown vector gains
$\Delta$, the $P$ equation of the former slack bus is kept, the Jacobian gains the column $-k$ (the derivative of
$P(\theta, |V|) - P^0 - k\Delta$ with respect to $\Delta$), and the angle reference stays at one bus. In
OpenLoadFlow, the balance type selects $k$ (proportional to $P_\text{max}$, to the target $P$, and so on).

**Octave + MATPOWER.**

```octave title="matpower_pf.m"
--8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/matpower_pf.m"
```

## Lab

!!! example "Lab: Newton–Raphson from scratch, then break it"

    Tools: Jupyter + NumPy + matplotlib + pandapower; optionally Octave + MATPOWER, pypowsybl and PowerWorld
    Simulator.

    1. Implement Newton–Raphson for a 5-bus grid (reuse [Chapter 6](../../part-1-math/ch06-newton-raphson/index.md) code
       and the $Y_\text{bus}$ of [Chapter 14](../ch14-component-models/index.md)), with the Jacobian of the worked example.
       Plot the mismatch norm per iteration on a log scale and confirm quadratic convergence.
    2. Run `pandapower.runpp` on the same data (or `runpf` in Octave) and compare voltage magnitudes and angles.
    3. Break convergence on purpose: scale the load until the solver fails, remove a reactive source, or open a
       line that islands a load. Record the iteration history each time and classify the failure.
    4. Compare flat start, DC-angle start and a start from a previous solution on a heavily loaded case: count
       iterations and check whether all starts converge to the same voltages.
    5. Optional: with `pypowsybl.network.create_ieee14()`, run `pypowsybl.loadflow.run_ac` with
       `Parameters(distributed_slack=False)` and with `distributed_slack=True`, and tabulate how generator active
       powers and the largest branch flows change.
    6. Optional (Windows): build the 5-bus grid of step 1 in the PowerWorld Simulator demo, with the same MVA
       base, solve it, and compare bus voltages and branch flows with your solver. The demo is limited to
       13 buses, so IEEE 14 does not fit. Then repeat one failure of step 3 and compare how Simulator reports it
       with your iteration history.

## Exercises

1. For a two-bus system (slack bus 1 at $1\angle 0$, PQ bus 2 with load $P + jQ$, line reactance $X$), write the
   power flow equations and the $2 \times 2$ Jacobian, and do two Newton iterations by hand for
   $X = 0.1$, $P = 1$, $Q = 0.5$ p.u.
2. *(GeoGebra)* Two-bus power transfer: with sliders for $|V_1|$, $|V_2|$, $X$ and $\delta$, plot
   $P = |V_1||V_2|\sin\delta / X$ and $Q_2 = (|V_1||V_2|\cos\delta - |V_2|^2)/X$. Find the maximum transfer and
   explain what happens to Newton–Raphson near $\delta = 90^\circ$.
3. *(Jupyter: NumPy)* Verify the polar Jacobian formulas of the worked example against central finite
   differences on IEEE 14, then time the dense and `scipy.sparse` versions.
4. *(Octave + MATPOWER)* Solve `case118` with `pf.alg` set to `NR`, `FDXB` and `GS`; tabulate iterations and time,
   and explain the ranking with the cost per iteration of each method.
5. Explain why a DC power flow can give a solution for a topology whose AC power flow diverges. List three
   physical causes and the symptom each leaves in the iteration history.
6. *(pypowsybl)* Repeat the distributed-slack comparison of the lab with `BalanceType.PROPORTIONAL_TO_GENERATION_P`
   and `BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX`. Which generators pick up the losses in each case?
