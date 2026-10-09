# Chapter 15. AC power flow

**Weeks:** 2.5 · **Semester:** 3 · **Prerequisites:** [Chapter 6](../part-1-math/ch06-newton-raphson.md), [Chapter 14](ch14-component-models.md)

!!! info "Why it matters for ToOp"

    The final word on every candidate topology is an AC N-1 analysis
    (`contingency/ac_loadflow_service/`, pandapower and pypowsybl runners). ToOp's AC-only metrics (voltage jump
    after a contingency, voltage-angle difference across open breakers, non-converging load flows) come from
    here, and so does the AC–DC mismatch correction applied to DC flows
    ([Chapter 17](ch17-sensitivity-factors-1.md)).

    Two AC settings shape everything downstream:

    - **Distributed slack.** ToOp's PowSyBl parameter sets (`grid_helpers/powsybl/loadflow_parameters.py`) use
      OpenLoadFlow with `distributed_slack=True`: `POWSYBL_LOADFLOW_PARAM_PF` balances with
      `BalanceType.PROPORTIONAL_TO_GENERATION_P_MAX`, while `CGMES_DISTRIBUTED_SLACK` and
      `UCTE_DISTRIBUTED_SLACK` use `BalanceType.PROPORTIONAL_TO_GENERATION_P` with
      `slackDistributionFailureBehavior = LEAVE_ON_SLACK_BUS`. `docs/dc_solver/loadflow_parameters.md` justifies
      the distributed slack by the smaller AC–DC mismatch it produces. The DC solver itself, in contrast, uses a
      single slack bus ([Chapter 17](ch17-sensitivity-factors-1.md)).
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
  [Chapter 6](../part-1-math/ch06-newton-raphson.md) on a real system).
- Gauss–Seidel; fast decoupled load flow (the $B'$ and $B''$ matrices, MATPOWER `makeB`).
- Reactive power limits and PV→PQ switching; voltage control.
- Distributed slack: participation factors and how they change the Jacobian; OpenLoadFlow balance types.
- Initialization (flat, DC, previous values); convergence failures and what they indicate (voltage collapse,
  islands, bad data); why the initialization can decide which solution you get.
- Tools: MATPOWER `runpf` and `mpoption` in Octave; `pandapower.runpp`; `pypowsybl.loadflow.run_ac`.

## Learning outcomes

- Implement a Newton–Raphson power flow for a 5-bus system and match pandapower's (or MATPOWER's) results.
- Write the four Jacobian blocks in polar form and check them against finite differences.
- Explain why an AC power flow can fail to converge for a topology that the DC model accepts.
- Compare single-slack and distributed-slack solutions of the same case and explain the difference in
  generator outputs and branch flows.

## Resources

- Free: Overbye ECEN 615 power flow lectures; MATPOWER manual (§4.1 AC power flow, Table 4-2 power flow
  options) <https://matpower.org/docs/MATPOWER-manual.pdf>; pandapower power flow tutorials
  <https://github.com/e2nIEE/pandapower/tree/master/tutorials> and AC power flow documentation
  <https://pandapower.readthedocs.io/en/latest/powerflow/ac.html>; PowSyBl OpenLoadFlow parameters
  <https://powsybl.readthedocs.io/projects/powsybl-open-loadflow/en/latest/loadflow/parameters.html>.
- Software: GNU Octave <https://octave.org/> with MATPOWER <https://matpower.org/>; GeoGebra
  <https://www.geogebra.org/classic>.
- Papers: Tinney & Hart, "Power flow solution by Newton's method", *IEEE Trans. PAS* 86(11), 1967; Stott &
  Alsaç, "Fast decoupled load flow", *IEEE Trans. PAS* 93(3), 1974.
- Textbooks:
  - Wood, Wollenberg & Sheblé, *Power Generation, Operation, and Control*, 3rd ed. (2014), Ch. 6: §6.8
    Newton–Raphson, §6.16 decoupled power flow, §6.17 Gauss–Seidel (verified against the publisher's table
    of contents). 1st and 2nd eds. **PRINT-DISABLED**; 3rd ed. **NOT ON IA**.
  - Power flow chapters of Grainger & Stevenson, Glover et al., Saadat ([Chapter 14](ch14-component-models.md)).
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

```python
import numpy as np

def injections(theta, vm, Y):
    v = vm * np.exp(1j * theta)
    return v * np.conj(Y @ v)          # S = [V] conj(Y V)

def jacobian_blocks(theta, vm, Y):
    v = vm * np.exp(1j * theta); i = Y @ v; E = np.diag(v / vm)
    dS_dth = 1j * np.diag(v) @ np.conj(np.diag(i) - Y @ np.diag(v))
    dS_dvm = np.diag(v) @ np.conj(Y @ E) + np.diag(np.conj(i)) @ E
    return dS_dth.real, dS_dvm.real, dS_dth.imag, dS_dvm.imag   # J11, J12, J21, J22
```

**Distributed slack.** Replace the slack bus's unknown $P$ by a scalar mismatch $\Delta$ shared by the
participating generators, $P_{g,i} = P_{g,i}^0 + k_i \Delta$ with $\sum_i k_i = 1$. The unknown vector gains
$\Delta$, the $P$ equation of the former slack bus is kept, the Jacobian gains the column $-k$ (the derivative of
$P(\theta, |V|) - P^0 - k\Delta$ with respect to $\Delta$), and the angle reference stays at one bus. In
OpenLoadFlow, the balance type selects $k$ (proportional to $P_\text{max}$, to the target $P$, and so on).

**Octave + MATPOWER.**

```octave
mpc = loadcase('case14');
mpopt = mpoption('pf.alg', 'NR', 'pf.enforce_q_lims', 1, 'verbose', 2, 'out.all', 0);
results = runpf(mpc, mpopt);
results.success            % 1 if converged
mpopt = mpoption(mpopt, 'pf.alg', 'FDXB');   % fast decoupled, XB version
results_fd = runpf(mpc, mpopt);
```

## Lab

!!! example "Lab: Newton–Raphson from scratch, then break it"

    Tools: Jupyter + NumPy + matplotlib + pandapower; optionally Octave + MATPOWER and pypowsybl.

    1. Implement Newton–Raphson for a 5-bus grid (reuse [Chapter 6](../part-1-math/ch06-newton-raphson.md) code
       and the $Y_\text{bus}$ of [Chapter 14](ch14-component-models.md)), with the Jacobian of the worked example.
       Plot the mismatch norm per iteration on a log scale and confirm quadratic convergence.
    2. Run `pandapower.runpp` on the same data (or `runpf` in Octave) and compare voltage magnitudes and angles.
    3. Break convergence on purpose: scale the load until the solver fails, remove a reactive source, or open a
       line that islands a load. Record the iteration history each time and classify the failure.
    4. Compare flat start, DC-angle start and a start from a previous solution on a heavily loaded case: count
       iterations and check whether all starts converge to the same voltages.
    5. Optional: with `pypowsybl.network.create_ieee14()`, run `pypowsybl.loadflow.run_ac` with
       `Parameters(distributed_slack=False)` and with `distributed_slack=True`, and tabulate how generator active
       powers and the largest branch flows change.

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
