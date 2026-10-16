# Chapter 19 – Solutions

Solutions of the Lab and the Exercises of [Chapter 19](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`metrics.py`](src/metrics.py) (the DC metrics in NumPy),
[`security57_toop.py`](src/security57_toop.py) (Lab Part A, ToOp), [`ac_n1_toop.py`](src/ac_n1_toop.py) (Lab Part B,
ToOp), [`backend_compare.py`](src/backend_compare.py) (Lab Part B, step 5), [`case7.py`](src/case7.py) (SciPy and
pandapower) and [`case7_toop.py`](src/case7_toop.py) (ToOp), [`metric_design.py`](src/metric_design.py)
(Exercises 1–4 and 6) and [`matpower_n1_metrics.m`](src/matpower_n1_metrics.m) (Exercise 5, Octave + MATPOWER).
Each module's `main()` prints the numbers quoted below; the ToOp numbers were produced with ToOp `ba1874e` on
CPU, the Octave ones with Octave 11.3 and MATPOWER 8.1. `python case7.py` prints every number of the Part C
answers and of Exercise 10 (a) and (c); `python case7_toop.py <folder>` prints those of Exercises 7–10 (b).
The notebooks [examples.ipynb](src/examples.ipynb), [lab.ipynb](src/lab.ipynb) (Part C),
[lab_toop.ipynb](src/lab_toop.ipynb) (Parts A and B) and [exercises.ipynb](src/exercises.ipynb) import these
modules.

## Lab, Part A: DC metrics from the full N-1 matrix

1. **Load IEEE 57.** `case57_data_powsybl(folder)` writes ToOp's IEEE 57 example in PowSyBl format and
   `load_grid(DirFileSystem(folder), pandapower=False, parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False))`
   preprocesses it. `stats` reports 72 nodes (57 buses plus the auxiliary nodes of the relevant stations), 82
   branches, 81 branch outages + 3 injection outages = 84 N-1 cases, 12 relevant substations, 127 actions and
   `has_double_limits=True`. `run_initial_loadflow` returns the unsplit
   `(overload_energy_n_0, overload_energy_n_1) = (454.2859, 3249.7961)` MW: this example is heavily overloaded
   on purpose.
2. **Full matrices.** `default_topology(solver_config, batch_size=4)` gives four unsplit topologies (every action
   slot holds the padding index). With `passthrough(lf) = (lf.n_0_matrix, lf.n_1_matrix)` defined once at module
   level, `run_solver_symmetric(topologies, None, None, dynamic_information, solver_config, passthrough)` returns
   `n_0` of shape (4, 1, 82) and `n_1` of shape (4, 1, 84, 82): topology, timestep, N-1 case, monitored branch;
   all 4 × 84 success flags are `True`. Defining the function inside a loop would make JAX retrace and
   recompile the solver on every call.
3. **Energies in NumPy.** `branch_limits.overload_weight` is 2.0 on the 64 lines and 1.0 on the 18
   transformers, and `max_mw_flow_n_1` is exactly twice `max_mw_flow` for every branch. With
   `metrics.overload_energy_n1(n_1[0], max_mw_flow_n_1, weight)` and the N-0 vector as a one-case matrix, NumPy
   gives 454.285915469 and 3249.796127648 MW; the differences to ToOp are $1 \cdot 10^{-13}$ and
   $4 \cdot 10^{-13}$ MW (`test_security57_toop.py` asserts $10^{-6}$).
4. **Double limits.** `metrics.double_limits` (90 % of the rating below 90 % loading, the flow itself between
   90 % and 100 %, the rating above) reproduces `max_mw_flow_limited` and `max_mw_flow_n_1_limited` exactly
   (largest difference 0.0). For the N-1 limits, 26 branches are in the lower regime (ratio limited/rating
   0.9), 8 in the dead zone (ratio equal to their loading) and 48 above 100 % (ratio 1). Plotted with the
   relative loading on the x axis and limited/rating on the y axis, the points lie on a horizontal line at 0.9
   up to $x = 0.9$, on the diagonal $y = x$ between 0.9 and 1.0, and on a horizontal line at 1.0 beyond: the
   three regimes of `compute_double_limits`.
5. **Real actions.** `topologies.action.at[1, 0].set(1).at[2, 0].set(5)`, rebuilt as
   `ActionIndexComputations(action=..., pad_mask=topologies.pad_mask)`, applies action 1 to the second topology
   and action 5 to the third. For the four topologies ToOp gives, through
   `functools.partial(aggregate_to_metric, branch_limits=..., reassignment_distance=..., n_relevant_subs=..., metric=...)`
   (one partial per metric, each built once):

    | Metric | unsplit | action 1 | action 5 | unsplit |
    |---|---|---|---|---|
    | `overload_energy_n_0` | 454.29 | 459.77 | 468.13 | 454.29 |
    | `overload_energy_n_1` | 3249.80 | 3465.16 | 3386.51 | 3249.80 |
    | `exponential_overload_energy_n_1` | 6841.80 | 6994.97 | 7171.36 | 6841.80 |
    | `critical_branch_count_n_1` | 48 | 53 | 49 | 48 |
    | `n0_n1_delta` | 0 | 0 | 0 | 0 |
    | `overload_energy_limited_n_0` | 454.29 | 461.24 | 469.00 | 454.29 |
    | `overload_energy_limited_n_1` | 3249.80 | 3496.37 | 3403.98 | 3249.80 |

    `numpy_metrics` reproduces all seven columns to $10^{-12}$. Both actions make this grid worse. `n0_n1_delta`
    is zero because the case57 generator writes no `*_n0_n1_max_diff_factor.npy` masks, so every factor is
    $-1$ and the penalty is disabled for every branch (`compute_n0_n1_max_diff` stores $-1$,
    `get_n0_n1_delta_penalty` zeroes those branches).

    ```python title="security57_toop.py"
    --8<-- "docs/part-3-power-systems/ch19-operational-security/src/security57_toop.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch19-operational-security/src){ target="_blank" } to see or download all the files of this chapter.

## Lab, Part B: AC N-1 and metrics from result tables

1. **The N-1 definition.** `data/grid_node_breaker/grid.xiidm` has 9 lines, no transformers, 5 buses in the
   bus-breaker view and 106 switches. One `MonitoredElement(id, type="LINE", kind="branch")` per line and one
   `MonitoredElement(id, type="BUS", kind="bus")` per bus (14 monitored elements), one
   `Contingency(id, elements=[GridElement(id, type="LINE", kind="branch")])` per line plus
   `Contingency(id="BASECASE", elements=[])`: 10 contingencies.
2. **Run.** `get_ac_loadflow_results(net, definition, job_id="lab", timestep=0, n_processes=1)` returns polars
   LazyFrames; `.collect().to_pandas()` gives 162 branch rows, 49 node rows, 9 `va_diff` rows and 10
   convergence rows, all `CONVERGED`.
3. **Metrics with pandas.** The MW limit behind a row is $|p| / \text{loading}$, so its overload is
   $|p| - |p|/\text{loading}$ (the same with the current `i`); per `(timestep, element)` the maximum over
   contingencies and sides, then the sum of the positive ones. Critical branches are the distinct
   `(timestep, element)` with `loading > 1` among the N-1 rows; `max_va_diff_n_1` is the largest $|\text{va\_diff}|$.

    | Metric | pandas | `compute_metrics` |
    |---|---|---|
    | `overload_energy_n_1` | 62.113903 MW | 62.113903 MW |
    | `overload_current_n_1` | 160.378002 A | 160.378002 A |
    | `critical_branch_count_n_1` | 3 | 3 |
    | `max_va_diff_n_1` | 0.485122° | 0.485122° |

    The worst rows are all in the L9 contingency: L3 at 191 % (66.7 MW), L1 and L2 at 143 %.
4. **The row count.** The formula gives $(2 \cdot 9 + 0 + 0) \cdot 10 = 180$ rows; the table has 162. The 18
   missing rows are the two sides of the outaged line in its own contingency (9 contingencies × 2 sides): an
   element that is out of service has no flow to report, and `compute_overload_column` would otherwise divide
   by a zero loading. A contingency that did not converge would remove all its rows as well (none here).

    ```python title="ac_n1_toop.py"
    --8<-- "docs/part-3-power-systems/ch19-operational-security/src/ac_n1_toop.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch19-operational-security/src){ target="_blank" } to see or download all the files of this chapter.

5. **pandapower against pypowsybl (optional).** `backend_compare.py` exports pandapower's `case14` with
   `to_mpc` after a DC power flow, loads the MATPOWER file in pypowsybl (17 lines and 3 transformers: the two
   branches with unit tap ratio become lines) and runs the same 20 single-branch outages in both engines. With
   pypowsybl's `distributed_slack=False` (one slack at bus 1, as pandapower's `ext_grid`) and
   `use_reactive_limits=False` (pandapower's `runpp` ignores generator Q limits unless `enforce_q_lims=True`),
   the 380 compared from-side flows differ by 0.0001 MW at the median, 0.0019 MW at the 90th percentile and
   0.0326 MW at most (outage of 10–11, branch 9–10: 9.033 against 9.001 MW); a histogram of the differences is
   a spike at zero with a tail below 0.04 MW. With reactive limits on, the differences reach 0.65 MW and
   pypowsybl drops 19 more rows, which is why ToOp's `test_backend_comparison.py` fixes its load flow
   parameters (`UNIFORM_SINGLE_SLACK`) before asking for $10^{-3}$ agreement. The remaining differences come
   from the transformer models and the per-unit conversions of the two engines. This step needs no ToOp.

    ```python title="backend_compare.py"
    --8<-- "docs/part-3-power-systems/ch19-operational-security/src/backend_compare.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch19-operational-security/src){ target="_blank" } to see or download all the files of this chapter.

## Lab, Part C: the price of security on the 7-bus case

1. **Binding constraints.** `binding(dispatch(range(11)).x)` returns two pairs: line 1–3 after the outage of
   line 1–2, and line 1–2 after the outage of line 1–3. Both lines carry exactly 240 MW. The cheap generator at
   bus 1 (capped at 250 MW) has only these two lines to leave the bus, so when one trips the other carries all of
   it, and the SCOPF holds it to the limit by moving 10 MW to the generator at bus 2.
2. **AC security.** The AC N-1 of the SCOPF dispatch peaks at 109.7 % (114.7 % for the plain OPF). Deriating the
   limits of the overloaded lines to (worst DC flow) × 100 / (worst AC loading) and solving again gives a SCOPF
   cost of $4,400/h, then $4,612/h, then $4,672/h, at which point the AC worst loading is at most 100 %
   (`ac_secure_dispatch()`). AC security costs $272/h more than DC security and $372/h more than the plain OPF.
3. **Line 1–3 limited to 150 MW.** The OPF stays at $4,300/h: its base-case flow on line 1–3 is 133 MW, below the
   new limit, and an OPF never looks at outages. The SCOPF costs $5,360/h with dispatch (146.3, 192.2, 1.5) MW.
   The outage of line 1–2 sends all of the generator at bus 1 through line 1–3, so that generator must stay
   below about 150 MW. Instead of 10 MW, about 104 MW move to the $20 generator at bus 2, which then reaches its
   own network limits (binding: line 1–3 after the outage of line 2–3, line 2–3 after the outage of line 2–5),
   and 1.5 MW come from the $35 generator at bus 7. The gap grows because the security constraint, not the
   price difference, sets the quantity moved. A remedial topology action does not help here for the same reason
   as in [Exercise 9](#exercise-9): the constraint is the cut at bus 1, and no split downstream changes it. Redispatch is the
   remedy.
4. **PowerWorld.** Expect the same dispatch and cost for a linear (DC) OPF and SCOPF: (250, 90, 0) MW at
   $4,300/h and (240, 100, 0) MW at $4,400/h. An AC OPF will cost a little more, because it includes losses and
   reactive flow; that is the same gap as in step 2. Differences in the limit handling (percent of MVA rating
   versus MW) show up as small dispatch changes.
5. **ToOp.** See [Exercise 7](#exercise-7) to [Exercise 10](#exercise-10).

## Exercise 1

The worked example has $f_\max = (100, 200, 50)$ MW and N-1 flows $(95, -210, 40)$ and $(120, -160, 55)$: overload
energy $20 + 10 + 5 = 35$ MW, critical branch count 3. All three branches are already critical, so the count
can only rise from a state where one of them is not; and the count can never move while the energy stays
exactly put: a branch becomes (or stops being) critical exactly when its worst-case overload $\max_c (|F_{cb}|
- \bar F_b)^+$ leaves (or reaches) zero, and that same quantity is the branch's term in the energy. What the
exercise shows is how *little* energy a count change needs. Start from a state where branch 3 is below its
limit (scale its two flows by 0.9: 36 and 49.5 MW; energy 30 MW, count 2) and raise the 49.5 MW to 50.001 MW:
the energy moves by $\varepsilon = 0.001$ MW, the count by one (`metric_design.energy_and_count`). The
opposite direction is easy: raise branch 1 in the second contingency from 120 to 140 MW and the energy goes
from 35 to 55 MW while the count stays 3. The change that moves neither metric: raise branch 1 in the *first*
contingency from 95 to 115 MW and nothing moves at all, because 115 MW is still below that branch's worst case
of 120 MW (the per-branch maximum hides every contingency but the worst).

What each objective prefers: under overload energy the optimizer trades a 20 MW relief on one branch against a
19.9 MW new overload elsewhere, and it is indifferent to how many branches carry the overload; under the
critical branch count it would rather concentrate 50 MW on one branch than spread 2 MW over three, and it gives
nothing for relieving an overloaded branch that stays overloaded. ToOp's weighted sum ($-(35 + 10 \cdot 3) =
-65$ in the chapter) is the compromise: the count term rewards healing a branch completely, the energy term
rewards every MW on the way there.

```python title="metric_design.py"
--8<-- "docs/part-3-power-systems/ch19-operational-security/src/metric_design.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch19-operational-security/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

Per branch and as functions of the loading $\ell = |F| / \bar F$ (`metric_design.penalty_curves`):

$$
\text{overload energy} = (\ell - 1)^+ \bar F, \qquad
\text{power law} = \left(\max(\ell, 1)^{1.5} - 1\right)\bar F, \qquad
\text{cumulative overload} = (\ell - 1)^+ .
$$

Plotted from 0 to 200 % loading for $\bar F = 100$ MW and $\bar F = 1\,000$ MW (two panels, one per metric, or one
panel per limit), the first two are flat at zero up to 100 % and then rise, linearly for the energy and
convexly for the power law, with the 1 000 MW curves ten times higher than the 100 MW ones; the cumulative
overload is the same curve for both limits. At 110 % and 150 % the values are 10 and 50 MW (energy), 15.4 and
83.7 MW (power law) for 100 MW, 100 and 500 MW, 153.7 and 837.1 MW for 1 000 MW, and 0.1 and 0.5 (cumulative
overload) for both. The cumulative overload is the capacity-blind metric that makes a 10 % overload on a
110 kV line as costly as on a 380 kV line; it is the only one of the three that does so, since the power law
scales with $\bar F$ like the energy (ToOp's `get_cumulative_overload_n_1_matrix` divides the overload matrix by
`max_mw_flow` and sums the per-branch maxima). The power law's extra is a bias against *deep* overloads: at
150 % it charges 1.67 times the energy, at 110 % 1.54 times.

## Exercise 3

Two branches rated 100 MW, unsplit flows (95, 60) MW (`metric_design.dead_zone_example`). `double_limits` gives
limited limits (95, 90): branch 1 is in the dead zone, so its limit becomes its current flow; branch 2 is below
90 %, so its limit becomes 90. The unsplit `overload_energy_limited_n_0` is $0$ ($95 \le 95$, $60 \le 90$). A
topology that only shifts 2 MW from branch 2 to branch 1, flows (97, 58), is overloaded nowhere against the
ratings (`overload_energy_n_0` = 0) but gets `overload_energy_limited_n_0` = 2 MW. That is the purpose of the
rule: a branch at 95 % is one forecast error, one AC correction or one reactive flow away from its rating,
and the DC solver that ranks topologies does not see those. The dead zone tells the optimizer "do not load
this branch any further" without declaring it overloaded, which would otherwise push the search towards
relieving a branch that is in fact fine, and the 90 % rule keeps lightly loaded branches from being filled up
to the brim.

## Exercise 4

`evaluate_acceptance` checks, in order, with the default thresholds 1.0, 0.95 and 1.1
(`metric_design.acceptance` reproduces the rule):

1. Convergence: non-converging cases of the candidate minus its disconnected branches, $2 - 1 = 1$, must be
   $\le 1.0 \times 2$: passes.
2. Overload energy: $0 \le 0.95 \times 0 = 0$: passes (zero is not an improvement, but it is not above zero).
3. Critical branches: $1 \le 1.1 \times 0 = 0$: **fails**. The candidate is rejected with the criterion
   `critical-branch-count`.

The rejection is right in substance (the candidate created an overloaded branch where there was none) but the
mechanism is fragile: a relative threshold around a zero baseline is an absolute threshold of zero, so a grid
that is N-1 secure to begin with can accept no candidate with the slightest new violation, and the 10 % margin
that the designers intended never exists. An absolute slack (allow at most $k$ new critical branches, or at
most $\delta$ MW of new overload energy) behaves the same whether the baseline is 0 or 100; a hybrid
$\max(1.1 \cdot x, x + 1)$ keeps the relative behaviour where the baseline is large. ToOp's own design notes
agree: with a zero baseline, "$1.1 \times 0 = 0$ rejects any increase" (chapter text). `acceptance_with_slack`
with one branch of slack accepts this candidate. The same rule accepts (94 MW, 11 branches) against (100 MW,
10 branches) and rejects (96 MW, 10) against (100 MW, 10) for the energy, which shows the intended behaviour
away from zero.

## Exercise 5

`matpower_n1_metrics('case30')` builds $H$ with `makePTDF`, the LODF with `makeLODF(mpc, H, true)` (NaN columns
for bridges), the base flows $F = HP$ in MW, limits $1.2\,|F|$ and the N-1 matrix $F_1(:, k) = F + L(:, k)\,F_k$
column by column. case30 has 41 branches, of which three are bridges (9–11, 12–13 and 25–26) and are skipped,
and one branch with zero base flow (9–11: bus 11 has neither load nor generator in MATPOWER's `case30`), whose
limit is therefore zero; a tolerance of $10^{-9}$ MW keeps round-off on that branch from counting as an
overload. With
every limit 20 % above its base flow, almost every single outage overloads something: the overload energy is
258.78 MW and 38 of the 41 branches are critical (all but the three bridges, which carry nothing new). The two
largest terms are the branches 8–28 (23.7 MW) and 6–28 (21.4 MW) after the outage of 6–8, and 1–2 after the
outage of 1–3 (12.5 MW).

Doubling the 30 MW load at bus 8 (limits kept at their original values) raises the overload energy to
499.13 MW and lowers the count to 37 (branch 21–22 is no longer critical). The worst contingency changes for
six branches, 2–5, 5–7, 4–12, 12–15, 12–16 and 16–17 (branches 5, 8, 15, 18, 19, 21): before, their worst
cases were the outages of 6–7 (branches 2–5 and 5–7), 12–16 (branch 4–12), 15–23 (branch 12–15) and 15–18
(branches 12–16 and 16–17); after, the outage of 4–6 (branch 7) is worst for five of them and the outage of
12–14 for 12–15. The extra 30 MW come from the slack at bus 1 and travel to bus 8 mainly over 1–2, 2–4, 4–6
and 6–8, so those lines are now the heavy ones: when 4–6 trips, its larger flow is rerouted over 2–5, 5–7 and
the 4–12 corridor and overloads them, whereas before the load increase the smaller rerouted amount was not the
worst thing that could happen to them. The branches directly on the new path
keep their worst contingency but their overloads grow (1–2 after 1–3 out: 12.5 → 42.5 MW; 8–28 and 6–28 after
6–8 out: 23.7 → 53.7 and 21.4 → 49.8 MW). This is the per-branch maximum at work: a load change does not only
raise overloads, it reshuffles which outage each branch has to fear.

```octave title="matpower_n1_metrics.m"
--8<-- "docs/part-3-power-systems/ch19-operational-security/src/matpower_n1_metrics.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch19-operational-security/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 6

The DC fitness is $-\sum_m w_m\,\text{metric}_m$. One objective that gives $-713$ is the chapter's own weight
pair, `overload_energy_n_1` with weight 1 and `critical_branch_count_n_1` with weight 10, at 213 MW and 50
branches: $213 + 10 \cdot 50 = 713$ (`metric_design.fitness_table`, a pandas table with one row per metric and a
`contribution` column). Many other combinations produce the same number (113 MW and 60 branches, or an extra
`n0_n1_delta` term), which is the first reason the number cannot be read back: a weighted sum has no inverse.
The AC fitness 12.4 is a single metric, `overload_energy_n_1` in MW from the AC N-1, with the opposite sign
convention (lower is better). The two are not comparable because: they have opposite signs; one is a weighted
composite and the other a single quantity; the DC value is computed on DC flows against (possibly double)
MW limits over the DC contingency list, the AC value on AC loadings (current, including reactive flow and
losses) with the limit of each row derived from its loading; and the AC stage reports it only for the
topologies that the DC stage proposed, while the DC optimizer compares its number across its whole
population. What can be compared is a *difference* within one stage: the DC fitness of two topologies, or the
AC overload energy of a candidate against the unsplit grid, which is exactly what `evaluate_acceptance` does.

## Exercise 7


`write_case` in the script writes the folder; the pass-through function returns
`(lf.n_0_matrix, lf.n_1_matrix)` and is defined once, at module level. `run_initial_loadflow` returns
`(overload_energy_n_0, overload_energy_n_1) = (0.0, 20.0)`, and the unsplit N-1 matrix gives, per outage,
the worst loaded line:

| Outage | Worst line | Loading |
|---|---|---|
| L1-2 | L1-3 | 104.2 % |
| L1-3 | L1-2 | 104.2 % |
| L2-3 | L1-3 | 71.8 % |
| L2-5 | L2-3 | 85.7 % |
| every other line | L1-3 or L2-5 | below 67 % |

The SciPy code gives the same 104.2 % (`n1_loading(dispatch().x).max()`). The two agree to the decimal
because both solve the same lossless DC equations, and the LODF update is exact, not an approximation.
The 20 MW follow from the definition: for each branch the worst contingency, then the sum over branches.
L1-3 carries 250 MW after the L1-2 outage (10 MW above its 240 MW), L1-2 carries 250 MW after the L1-3
outage (another 10 MW), and no other branch exceeds its limit in any contingency: 10 + 10 = 20 MW. Under
the "worst contingency per timestep" wording of `metrics.md` (see "Documentation vs code" above) the value
would be 10 MW; this case shows that the code sums per branch.

## Exercise 8


Choosing the set $B$ of branches that go to the second busbar with $2 \le |B| \le d - 2$ gives
$2^d - 2 - 2d$ configurations; each split appears twice ($B$ and its complement), so a substation with $d$
branches has $2^{d-1} - d - 1$ splits: 3 for $d = 4$, 10 for $d = 5$, 25 for $d = 6$ and none for $d = 3$.
Buses 3, 5 and 6 have four lines each, bus 4 has three (L3-4, L4-5, L4-6), so the prediction is
$3 \times (3 + 1) = 12$ actions. ToOp reports `n_actions_per_sub = [4, 4, 4]` for buses 3, 5 and 6, and
bus 4 is not in `relevant_nodes` at all. In the code, `enumerate_branch_actions_for_sub`
(`dc_solver/preprocess/action_set.py`) asserts $d \ge 4$ (smaller substations are dropped earlier),
enumerates $2^{d-1}$ configurations (the mirror image is not generated), removes those that isolate a
single branch (`exclude_isolations`) and then those that leave fewer than two non-bridge branches on a
busbar (`filter_splits_by_bridge_lookup`; no line of this grid is a bridge, so nothing more goes). Decoded,
the twelve actions are:

| Action | Substation | Branches on a busbar of their own |
|---|---|---|
| 0, 4, 8 | bus 3, 5, 6 | none (unsplit) |
| 1, 2, 3 | bus 3 | L3-6 + L1-3; L3-4 + L1-3; L3-4 + L3-6 |
| 5, 6, 7 | bus 5 | L5-7 + L2-5; L5-6 + L2-5; L5-6 + L5-7 |
| 9, 10, 11 | bus 6 | L3-6 + L4-6; L6-7 + L4-6; L6-7 + L3-6 |

The load of a substation stays with the branches that are not moved (`inj_actions` is all `False` here).

## Exercise 9


Kirchhoff's current law at bus 1: with L1-2 out, L1-3 is the only branch left at bus 1, so it carries the
bus's net injection, 250 MW of generation and no load, whatever the impedances or the topology downstream:
250 / 240 = 104.2 %. The same holds for L1-2 after the L1-3 outage. A split at buses 3, 5 or 6 changes how
the 250 MW spread beyond bus 3, never how they leave bus 1. ToOp confirms it:

| Action | Split | Largest N-0 change | Worst N-1 loading |
|---|---|---|---|
| 0, 4, 8 | unsplit | 0 MW | 104.2 % (L1-3 after L1-2 out) |
| 1 | bus 3: L3-6 + L1-3 | 44 MW | 125.0 % (L3-6 after L1-2 out) |
| 2 | bus 3: L3-4 + L1-3 | 31 MW | 125.0 % (L3-4 after L1-2 out) |
| 3 | bus 3: L3-4 + L3-6 | 137 MW | islanded (solver flag `False`, `nan`) |
| 5 | bus 5: L5-7 + L2-5 | 66 MW | 126.5 % (L2-3 after L1-3 out) |
| 6 | bus 5: L5-6 + L2-5 | 73 MW | 109.3 % (L2-3 after L1-3 out) |
| 7 | bus 5: L5-6 + L5-7 | 13 MW | 104.2 % |
| 9 | bus 6: L3-6 + L4-6 | 38 MW | 104.2 % |
| 10 | bus 6: L6-7 + L4-6 | 1 MW | 104.2 % |
| 11 | bus 6: L6-7 + L3-6 | 35 MW | 104.2 % |

No action gets below 104.2 %, four add a second overload, and action 3 turns line 2–5 into a bridge: the
busbar with L3-4 and L3-6 has no injection and no other link to buses 1 and 2, so already in the base case
L2-5 carries the whole 260 MW of load of buses 4–6 (108 % of its limit), and its outage islands buses 4–7
without generation. The `action_set_filter_bsdf_lodf` option, switched off in `write_case`, removes such
actions from the set ([Chapter 22](../ch22-import-preprocessing/index.md)). The only remedy is redispatch: the SCOPF
moves 10 MW from the $10 generator at bus 1 to the $20 one at bus 2, the $100/h of the worked example. In
general, a bus split redistributes flow among parallel paths; it cannot change the flow through a cut whose
remaining branches all lie on one side of a fixed injection.

## Exercise 10


(a) The old SCOPF dispatch loads L3-6 to 111.5 % after the L2-5 outage. The SCOPF with the tighter limit
costs $4,739/h with dispatch (240, 77.4, 22.6): the $35 generator at bus 7 must run, and the binding
constraints are now (L1-2 out, L1-3), (L1-3 out, L1-2) and (L2-5 out, L3-6). That is $339/h more than the
original SCOPF, the redispatch an operator would like to avoid.

(b) ToOp: the unsplit grid has `overload_energy_n_1 = 11.5 MW` (111.5 MW on a 100 MW line). Two of the
twelve actions clear every N-1 overload, both at bus 6: action 11, L3-6 + L6-7 on their own busbar, brings
L3-6 down to 60.2 MW, and action 9, L3-6 + L4-6, to 76.8 MW; the worst loading is then exactly 100.0 %, on
the lines 1–2 and 1–3 that the SCOPF had left at their limits. The third bus-6 split (L6-7 + L4-6) leaves
109.0 %, and every bus-3 and bus-5 split is worse than doing nothing (104.7 % to 240 %, or islanding).

(c) `split_bus(pandapower_net(pg, lim), 6, ['L3-6', 'L6-7'])` followed by `pandapower_n1` gives the same
60.2 MW on L3-6 and a worst DC loading of 100.0 %.

(d) After the L2-5 outage the 260 MW of load at buses 4, 5 and 6 must all come from bus 3 through the cut
{L3-4, L3-6}. The total through the cut is fixed by KCL, as in Exercise 9, but its split between the two
lines is not: it follows the impedances of the parallel paths behind the cut. Unsplit, L3-6 feeds bus 6
directly and bus 5 through L5-6 ($x = 0.14 + 0.10$). Paired with L6-7, whatever enters through L3-6 must
continue to bus 7 and reach bus 5 through L5-7 ($x = 0.14 + 0.12 + 0.15$), a much longer path than
L3-4 → L4-5 ($0.20$) or L3-4 → L4-6 ($0.22$), so most of the flow moves to L3-4 (199.8 MW of 200 MW after
the same outage). A split helps when the overloaded line has a parallel path downstream that the split
can lengthen or shorten; it is powerless when the line is the only path, as at bus 1
([Chapter 28](../../part-4-optimization/ch28-topology-optimization/index.md)). The AC check shows the price of the
thin DC margin: the split topology's worst AC N-1 loading is 116 % on L3-4, so this remedial action passes
ToOp's DC stage and fails its AC validation, which is the reason the two stages exist.
