# Chapter 15 – Solutions

Solutions of the Lab and the Exercises of [Chapter 15](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`ac_jacobian.py`](src/ac_jacobian.py) (the Jacobian blocks of the worked
example), [`nr_power_flow.py`](src/nr_power_flow.py) (the five-bus Newton–Raphson of the Lab, steps 1–4),
[`powsybl_slack.py`](src/powsybl_slack.py) (Lab step 5 and Exercise 6), [`two_bus.py`](src/two_bus.py)
(Exercises 1 and 2), [`jacobian_timing.py`](src/jacobian_timing.py) (Exercise 3) and
[`pf_algorithms.m`](src/pf_algorithms.m) (Exercise 4, Octave + MATPOWER). Every `main()` prints the numbers quoted
below; the timings are from one laptop and vary from run to run. The Lab steps are also in
[lab.ipynb](src/lab.ipynb) and the exercise statements in [exercises.ipynb](src/exercises.ipynb).

## Lab

1. **Newton–Raphson for a five-bus grid.** `CASE5` is a five-bus grid in MATPOWER column order: slack bus 1 at
   1.06 pu, PV bus 2 at 1.045 pu generating 40 MW, loads 20 + j10, 45 + j15, 40 + j5 and 60 + j10 MVA at buses 2
   to 5, and seven lines with $r/x = 1/3$ and line charging. `make_ybus` builds $Y_\text{bus}$ from the branch table
   (series admittance, half the charging at each end, taps and bus shunts), `specified` returns the bus types and
   the injections $P^\text{spec}, Q^\text{spec}$ in pu, and `newton_pf` iterates with the four blocks of
   `ac_jacobian.jacobian_blocks`, keeping the $P$ rows of the non-slack buses and the $Q$ rows of the PQ buses.
   From a flat start the mismatch infinity norm goes $4.9 \times 10^{-1} \to 3.3 \times 10^{-2} \to
   7.2 \times 10^{-5} \to 5.1 \times 10^{-10}$ pu: the exponent doubles at each step, which is the quadratic
   convergence of [Chapter 6](../../part-1-math/ch06-newton-raphson/index.md). Three iterations reach $10^{-8}$. The solution
   is $\theta = (0, -2.768, -4.977, -5.309, -6.129)^\circ$ and $|V| = (1.06, 1.045, 1.0223, 1.0215, 1.0156)$ pu.
   On a log scale the four points lie on a curve that bends down, not on a straight line: a straight line
   would be linear convergence.

    ```python title="nr_power_flow.py"
    --8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/nr_power_flow.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch15-ac-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

2. **pandapower on the same data.** `pandapower_net` builds the same grid with `create_line_from_parameters`:
   the per-unit branch data become ohms through $Z_\text{base} = V_\text{base}^2 / S_\text{base} = 230^2/100 =
   529\ \Omega$, and the line charging $b$ becomes a capacitance $C = b / (Z_\text{base}\, 2\pi f)$ in nF. `runpp`
   with `tolerance_mva=1e-10` agrees with step 1 to $5.5 \times 10^{-11}$ pu in magnitude and
   $1.3 \times 10^{-9}$ degrees in angle; `test_nr_power_flow.py` also checks that `make_ybus` equals pandapower's
   internal `Ybus` entry by entry. With Octave, save `CASE5` as a case file and call `matpower_pf` (worked
   example) on it; the voltages must agree to the same precision (not run here).
3. **Breaking convergence.** Three failures, three signatures:
    - *Loads scaled up* (`max_loading`): with every load multiplied by 5.20 the iteration still converges, in
      7 iterations, with the lowest voltage at 0.643 pu. At 5.25 it never does: the mismatch falls from 3.0 to
      $8.7 \times 10^{-3}$ in five iterations and then wanders between $10^{-2}$ and $10^{-1}$ for the remaining
      fifteen, with $|V| = (1.06, 1.045, 0.618, 0.622, 0.578)$ at iteration 20. This is the nose of the P–V
      curve: beyond the maximum loadability the equations have no solution, so Newton has nothing to converge to
      and the iterates oscillate around the point where the two solution branches merged. The Jacobian becomes
      singular exactly at the nose (Exercise 2).
    - *A reactive source removed* (`pv_to_pq(CASE5, 2)`): the generator at bus 2 no longer holds 1.045 pu and
      injects $Q = 0$. The nominal case still converges (4 iterations, $|V_2| = 1.032$ pu), but the largest load
      multiple drops from 5.20 to 2.95. Symptom: convergence on light load with low voltages and an earlier
      nose; near the limit the $Q$ rows of the mismatch dominate.
    - *An islanded load* (`without_branches(CASE5, [4, 6])`, lines 2–5 and 4–5 removed): bus 5 keeps its load but
      has no path to a voltage source. The rows and columns of bus 5 in the reduced Jacobian are zero, so
      `numpy.linalg.solve` raises `LinAlgError: Singular matrix` at the first iteration. Symptom: no history at
      all, a singular or ill-conditioned matrix. Production solvers detect islands first
      ([Chapter 3](../../part-1-math/ch03-graph-theory/index.md)) and either drop them or give each one a slack.
4. **Initializations.** On the case with the loads scaled by 5.20, the flat start and the DC-angle start both
   need 7 iterations, and the solution of the case scaled by 5.10 needs 4. All three end at the same voltages,
   $|V| = (1.06, 1.045, 0.6696, 0.6741, 0.6427)$ pu. Here the DC angles do not help, because the angles are not
   the problem (they stay below 30°): it is the voltage magnitudes, which the DC start leaves at 1 pu, that are
   far from the 0.64–0.67 pu solution. A previous solution gets both right and halves the work. On a heavily
   loaded case there is a second, low-voltage solution of the same equations; a start below the nose would
   converge to it, which is why `PREVIOUS_VALUES` and `DC_VALUES` are tried in order on import
   (`find_converging_loadflow_params`).
5. **Distributed slack with pypowsybl.** On the stock IEEE 14 nothing changes between
   `Parameters(distributed_slack=False)` and `True`: the generator targets (232.4 + 40 MW) already include the
   13.4 MW of losses, so the remaining slack mismatch is $-0.007$ MW and there is nothing to distribute. With
   every load raised by 10 % (`compare(1.1)`) the picture is:

    | Generator (target) | single slack | proportional to P | proportional to Pmax |
    |---|---|---|---|
    | B1-G (232.4 MW) | 232.4 | 257.27 | 246.97 |
    | B2-G (40 MW) | 40.0 | 44.28 | 54.57 |
    | B3-G, B6-G, B8-G (0 MW) | 0 | 0 | 0 |
    | slack mismatch left | 29.15 MW | $-0.27$ MW | $-0.91$ MW |
    | largest flow change vs single slack | – | 3.8 MW (L1-2) | 12.9 MW (L1-2) |

    With a single slack OpenLoadFlow leaves the 29.1 MW on the slack bus and reports it as
    `slack_bus_results[0].active_power_mismatch`; the slack generator's `p` keeps its target. With the slack
    distributed, the 29.1 MW go to the generators in the ratio 232.4 : 40 (`PROPORTIONAL_TO_GENERATION_P`) or
    equally (`PROPORTIONAL_TO_GENERATION_P_MAX`, both units have `max_p = 9999` MW); the three synchronous
    condensers with a 0 MW target never participate. The largest base-case flow, on line 1–2, is 177.4 MW with a
    single slack, 173.6 MW and 164.5 MW with the two distributed settings: how the losses are shared changes the
    flows by up to 13 MW on a 260 MW system, which is the AC–DC mismatch argument of ToOp's
    `loadflow_parameters.md`.

    ```python title="powsybl_slack.py"
    --8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/powsybl_slack.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch15-ac-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

6. **PowerWorld Simulator (optional, Windows).** Not run here. Method: create the five buses with the 230 kV
   nominal voltage, enter each line with its per-unit $R$, $X$ and $B$ (Simulator takes per-unit values on the
   system base, 100 MVA), set bus 1 as the slack at 1.06 pu, the bus 2 generator at 40 MW with 1.045 pu voltage
   setpoint and unlimited reactive power, the loads as constant power, and *Single Solution – Full Newton*.
   Expected outcome: the bus voltages and angles of step 1 to four decimals, and the line flows of
   `net.res_line` to 0.1 MW. For the failure of step 3, scale the loads with *Load Scaling* to 525 %: Simulator
   stops after its maximum number of iterations and reports the largest mismatch and its bus, the same
   stalling of the mismatch that your history shows; for the islanded load it removes the island before
   solving (the bus turns grey) and reports it in the message log, where your solver raised on the singular
   Jacobian.

## Exercise 1

With $V_1 = 1\angle 0$, $V_2 = v\angle\delta$ and $Y_\text{bus} = \begin{bmatrix} -j/X & j/X \\ j/X & -j/X
\end{bmatrix}$ ($G = 0$, $B_{21} = 1/X$, $B_{22} = -1/X$), the injections at bus 2 from the general formulas of the
worked example are

$$
P_2 = \frac{v}{X}\sin\delta, \qquad Q_2 = \frac{v^2 - v\cos\delta}{X},
$$

and the specified values are $P_2^\text{spec} = -P$, $Q_2^\text{spec} = -Q$ (a load). The Jacobian is

$$
J = \begin{bmatrix} \partial P_2/\partial\delta & \partial P_2/\partial v \\
\partial Q_2/\partial\delta & \partial Q_2/\partial v \end{bmatrix}
= \frac{1}{X}\begin{bmatrix} v\cos\delta & \sin\delta \\ v\sin\delta & 2v - \cos\delta \end{bmatrix}.
$$

Flat start $\delta = 0$, $v = 1$, with $X = 0.1$, $P = 1$, $Q = 0.5$:

- *Iteration 1.* $P_2 = Q_2 = 0$, mismatch $(-1, -0.5)$, $J = \operatorname{diag}(10, 10)$, step
  $(-0.1, -0.05)$: $\delta = -0.1$ rad ($-5.730^\circ$), $v = 0.95$.
- *Iteration 2.* $P_2 = 9.5\sin(-0.1) = -0.9484$, $Q_2 = (0.9025 - 0.95\cos 0.1)/0.1 = -0.4275$, mismatch
  $(-0.0516, -0.0725)$, $J = \begin{bmatrix} 9.4525 & -0.9983 \\ -0.9484 & 9.0500 \end{bmatrix}$, step
  $(-0.0064, -0.0087)$: $\delta = -0.1064$ rad ($-6.095^\circ$), $v = 0.9413$.

The exact solution, from $v^4 + (2QX - 1)v^2 + X^2(P^2 + Q^2) = 0$ (`exact_solution`), is $v = 0.9412$,
$\delta = -0.1064$ rad ($-6.099^\circ$): two iterations are within $10^{-4}$. The mismatch norms of the first five
iterations, $1.0,\ 7.2 \times 10^{-2},\ 8.9 \times 10^{-4},\ 1.3 \times 10^{-7},\ 2.9 \times 10^{-15}$, double
their exponent each time.

```python title="two_bus.py"
--8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/two_bus.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch15-ac-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

GeoGebra construction (not run here): four sliders `V1`, `V2` in $[0.9, 1.1]$, `X` in $[0.05, 0.5]$ and `d` in
$[0°, 180°]$; the functions `P(x) = V1 V2 sin(x) / X` and `Q2(x) = (V1 V2 cos(x) - V2²) / X` plotted against
the angle, and the point `(d, P(d))` moving on the curve. $Q_2$ as written in the statement is the reactive power
*received* at bus 2 (minus the injection of Exercise 1). The numbers below are from `two_bus.transfer` with
$|V_1| = |V_2| = 1$, $X = 0.1$:

| $\delta$ | 0° | 15° | 30° | 45° | 60° | 75° | 90° |
|---|---|---|---|---|---|---|---|
| $P$ (pu) | 0 | 2.588 | 5.000 | 7.071 | 8.660 | 9.659 | 10.000 |
| $Q_2$ (pu) | 0 | $-0.341$ | $-1.340$ | $-2.929$ | $-5.000$ | $-7.412$ | $-10.000$ |

The maximum transfer is $P_\max = |V_1||V_2|/X = 10$ pu at $\delta = 90^\circ$ (the slider past 90° brings $P$
down again: the same power is reached twice, once on each side). At that point the line absorbs
$|V_1||V_2|/X$ of reactive power. The derivative $\partial P/\partial\delta = |V_1||V_2|\cos\delta / X$ is
$5.0$ pu/rad at 60°, $0.87$ at 85°, $0.17$ at 89° and $0$ at 90°: the $P$–$\delta$ row of the Jacobian vanishes,
the matrix is singular and the Newton step $\Delta\delta = \Delta P / (\partial P/\partial\delta)$ blows up. For a
specified $P$ slightly below $P_\max$ the two solutions ($\delta$ and $180^\circ - \delta$) are close, the
Jacobian is nearly singular, and Newton converges slowly or jumps between them; for $P > P_\max$ there is no
solution and the mismatch stalls, as in Lab step 3.

## Exercise 3

`jacobian_timing.case_state` takes the converged state and the sparse `Ybus` of pandapower's `case14` and `case118`
(from `net._ppc['internal']`). `finite_difference_error` compares every column of the four blocks with central
differences of `injections` ($h = 10^{-6}$): the largest difference is $7.7 \times 10^{-9}$ on IEEE 14 and
$2.5 \times 10^{-8}$ on IEEE 118, the round-off level of a central difference, so the formulas are right.
`jacobian_blocks_sparse` is the same matrix expression with `scipy.sparse.diags` and a CSR `Ybus`. Timings (best of
three, this laptop):

| Case | buses | nonzeros of $Y_\text{bus}$ | dense | sparse |
|---|---|---|---|---|
| IEEE 14 | 14 | 54 | 0.030 ms | 0.79 ms |
| IEEE 118 | 118 | 476 | 0.98 ms | 0.88 ms |

On 14 buses the dense version is 25 times faster: the matrices have 196 entries and the sparse machinery's
fixed cost dominates. At 118 buses the two break even, and beyond that the dense cost grows with $n^2$ memory
and $n^3$ products while the sparse one grows with the number of branches. The real saving of sparsity is in the
linear solve of each iteration, not in building the blocks.

```python title="jacobian_timing.py"
--8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/jacobian_timing.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch15-ac-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`pf_algorithms` runs `runpf` with `pf.alg` set to `NR`, `FDXB` and `GS` at `pf.tol = 1e-8` and returns
`results.iterations` and `results.et`. Octave 11.3 with MATPOWER 8.1, after one warm-up call (the first `runpf` of a
session loads MATPOWER's classes and is several times slower):

| Case | NR | FDXB | GS |
|---|---|---|---|
| `case14` | 2 iterations, 0.098 s | 6 iterations, 0.184 s | 103 iterations, 0.355 s |
| `case118` | 3 iterations, 0.120 s | 8 iterations, 0.195 s | 2111 iterations, 10.4 s |

Gauss–Seidel needs `pf.gs.max_it` raised to 5000 on `case118` (2111 iterations are more than the default limit).
The ranking follows from the cost per iteration against the number of iterations. Newton–Raphson converges
quadratically (2–3 iterations) but each iteration builds the Jacobian and factorizes a new $(2n - n_{PV} - 2)$
matrix. The fast decoupled method factorizes $B'$ and $B''$ once, then does two cheap forward–backward
substitutions per iteration, but its convergence is linear (the decoupling $\partial P/\partial|V| \approx 0$,
$\partial Q/\partial\theta \approx 0$ is an approximation of the Jacobian), hence 6–8 iterations. Gauss–Seidel
costs one sweep through the buses per iteration, no factorization at all, and converges linearly with a rate
that worsens with the size of the system: 103 iterations on 14 buses, 2111 on 118. On small cases the wall time
is dominated by MATPOWER's setup, which is why FDXB does not beat NR here; its advantage shows in repeated
solves of large cases (contingency analysis), where the factorizations are reused.

```octave title="pf_algorithms.m"
--8<-- "docs/part-3-power-systems/ch15-ac-power-flow/src/pf_algorithms.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch15-ac-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

The DC power flow $P = B_\text{bus}\theta$ is linear: for any connected topology (with the slack in the
component) $\tilde B$ is a nonsingular grounded Laplacian ([Chapter 16](../ch16-dc-power-flow/index.md)), so there is
exactly one solution for every injection vector, however large. The AC equations are nonlinear and have a
solution only inside a bounded region of the injection space; outside it Newton–Raphson has nothing to find.
Three physical causes, with the symptom each leaves in the mismatch history (all three appear in Lab step 3):

1. **Active power beyond the transfer limit.** A corridor asked to carry more than $|V_f||V_t|/X$ (Exercise 2),
   because a topology change lengthened the path or a line outage left one line with the load of two. The DC
   model happily reports a flow of 150 % of the limit; in AC the $P$–$\delta$ curve has no point there. Symptom:
   the mismatch decreases for a few iterations, then stalls or oscillates around $10^{-2}$–$10^{-1}$ while
   angles across the corridor approach 90° (Lab 3, loads × 5.25).
2. **Reactive power deficit, voltage collapse.** Loads with reactive demand far from any voltage-controlled
   bus, after losing a generator, a shunt capacitor or a line that carried reactive support. The DC model
   ignores $Q$ and fixes $|V| = 1$, so it sees nothing. Symptom: the $Q$ rows dominate the mismatch, the
   voltage magnitudes sink iteration after iteration (0.6 pu and below), PV buses hit their reactive limits and
   switch to PQ, sometimes flip-flopping between the two types; the limit of the loading comes much earlier
   than with the reactive source present (Lab 3, bus 2 without voltage control: 2.95 instead of 5.20).
3. **Topology or data that make the Jacobian singular or ill-conditioned.** An island without a generator (the
   DC solve fails too, but only if the island is noticed), an island with generation but no slack, a branch with
   near-zero reactance, a wrong tap or a load entered in the wrong unit. Symptom: a singular-matrix error or a
   huge first step (voltages of 10 pu, angles of hundreds of degrees) followed by divergence, visible from the
   first iteration (Lab 3, bus 5 islanded). Note that ToOp's DC solver accepts the topology and the AC
   validation rejects it, which is what the "non-converging load flows" metric of the AC stage counts.

## Exercise 6

Same run as Lab step 5 (`powsybl_slack.compare(1.1)`), reading the pick-up above target per generator:

| Balance type | B1-G | B2-G | B3-G, B6-G, B8-G |
|---|---|---|---|
| `PROPORTIONAL_TO_GENERATION_P` | $+24.87$ MW | $+4.28$ MW | 0 |
| `PROPORTIONAL_TO_GENERATION_P_MAX` | $+14.57$ MW | $+14.57$ MW | 0 |

With `PROPORTIONAL_TO_GENERATION_P` the 29.1 MW of extra load and losses are shared in proportion to the active
power targets, 232.4 : 40 = 5.81 : 1, so the large unit at bus 1 takes 85 % of it. With
`PROPORTIONAL_TO_GENERATION_P_MAX` the weights are the `max_p` values, 9999 MW for every generator in
pypowsybl's IEEE 14, so the two generating units take half each. In both cases the synchronous condensers at
buses 3, 6 and 8, which have a 0 MW target, pick up nothing: OpenLoadFlow leaves units that produce no active
power out of the distribution. This is why ToOp's CGMES and UCTE parameter sets, with `PROPORTIONAL_TO_GENERATION_P`,
make the AC flows follow the dispatch pattern, while the `P_MAX` variant of `POWSYBL_LOADFLOW_PARAM_PF` moves
more of the balance to the units with the largest ratings whatever their current output.
