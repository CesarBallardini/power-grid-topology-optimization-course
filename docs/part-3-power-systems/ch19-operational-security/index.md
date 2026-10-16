# Chapter 19. Operational security and performance metrics

**Weeks:** 2.5 · **Semester:** 4 · **Prerequisites:** [Chapter 15](../ch15-ac-power-flow/index.md), [Chapter 17](../ch17-sensitivity-factors-1/index.md)

!!! info "Why it matters for ToOp"

    ToOp's objective is built from security metrics, and every design choice in those metrics changes which
    topology wins:

    - **DC metrics** (`docs/topology_optimizer/metrics.md`, `dc_solver/jax/aggregate_results.py`): overload
      energy over N-0 and N-1 cases, its power-law ("exponential") variant, critical branch count, the
      `_limited` variants that use double limits, the N-0→N-1 flow-change penalty `n0_n1_delta`, the
      cross-coupler flow penalty and the busbar outage penalty.
    - **Contingencies and monitored elements** are declared in `interfaces/nminus1_definition.py`.
    - **Fitness signs differ between stages**: the DC optimizer maximizes a negated weighted sum
      (`optimizer/dc/genetic_functions/scoring_functions.py`), while the AC stage reports overload energy
      directly and accepts or rejects a topology with multiplicative thresholds against the unsplit grid
      (`optimizer/ac/scoring_functions.py`, `optimizer/interfaces/messages/ac_params.py`).
    - **AC N-1 analysis** runs in pandapower or PowSyBl (`contingency/ac_loadflow_service/`), with metrics
      recomputed from result tables (`contingency/ac_loadflow_service/compute_metrics.py`).

    If you cannot recompute a metric by hand from an N-1 matrix, you cannot debug an optimizer that "improves"
    the wrong thing.

## Topics

- **Security criteria.** N-0, N-1 and N-k; contingency lists, monitored elements, exceptional contingencies
  (multi-outages, busbar outages). SO GL Articles 25 and 32–35 (operational security limits, power flow limits,
  contingency lists, contingency analysis and handling) and Article 72 (operational security analysis in
  operational planning). ToOp's `Nminus1Definition`: contingencies are lists of `GridElement`, monitored
  elements are `MonitoredElement`, and a contingency with no elements is the base case.
- **Thermal limits.** Permanent and temporary admissible transmission loading (PATL/TATL); transitory admissible
  overloads (SO GL Art. 25). Current limits become MW limits through $P = \sqrt{3}\,V_\text{nom} I$
  (revisits [Chapter 10](../../part-2-circuits/ch10-three-phase-per-unit/index.md)): ToOp's `get_p_max` reads the
  `permanent_limit` and `N-1` operational limits of each branch (`dc_solver/preprocess/powsybl/powsybl_helpers.py`).
  In AC, overload in MW is derived from the current loading, $|p| - |p|/\text{loading}$
  (`compute_overload_column` in `contingency/ac_loadflow_service/compute_metrics.py`), so the effective MW limit
  moves with voltage and power factor.
- **Metric semantics** (DC, for a flow matrix $F \in \mathbb{R}^{T \times C \times B}$ over timesteps,
  contingencies and monitored branches, limits $\bar F_b$ and optional weights $w_b$):
    - Overload energy: $\sum_{t}\sum_{b} w_b \max_{c}\,\big(|F_{tcb}| - \bar F_b\big)^+$. The maximum is taken
      **per branch** over contingencies, so different branches may have different worst contingencies.
    - "Exponential" overload energy is a **power law**, not an exponential:
      $\sum_{t}\sum_{b} w_b \max_c \big(\max(|F_{tcb}|/\bar F_b,\,1)^{\alpha} - 1\big)\bar F_b$ with
      $\alpha = 1.5$.
    - Critical branch count: branches overloaded in at least one contingency, counted in the **worst timestep**.
    - Cumulative overload (relative, capacity-blind) and underload energy (debugging only).
    - `n0_n1_delta`: $\sum_t\sum_b \big(\max_c (|F^{(1)}_{tcb}| - |F^{(0)}_{tb}|)^+ - D_b\big)^+$, where the allowed
      delta $D_b$ is the **unsplit** grid's worst delta times a per-branch factor from the
      `*_n0_n1_max_diff_factor.npy` masks; a negative factor disables the branch
      (`compute_n0_n1_max_diff`, `get_n0_n1_delta_penalty`). It keeps contingencies from dumping flow into
      distribution grids.
    - Cross-coupler flow penalty: MW above the coupler limit of each split station
      (`get_cross_coupler_flow_penalty`); it pushes the optimizer towards symmetric splits.
    - Busbar outage penalty relative to the baseline: overload increase plus a penalty per lost successful
      busbar outage case (`get_busbar_outage_penalty` in `dc_solver/jax/busbar_outage.py`); negative values
      (improvements) are kept unless `clip_bb_outage_penalty=True`.
    - The solver's sparse output keeps only the worst `number_most_affected` N-1 results
      (`SolverConfig` in `dc_solver/jax/types.py`); the full matrix needs `run_solver_symmetric`.
- **Double limits.** From the unsplit grid's worst flows, `compute_double_limits` sets the `_limited` limit to
  $0.9\,\bar F$ for branches loaded below 90 %, to the **current flow** for branches between 90 % and 100 %
  (a dead zone: do not load further, but do not count as overloaded), and to $\bar F$ above 100 %. It is on by
  default (`double_limit_n0`, `double_limit_n1` in `PreprocessParameters`) but only affects the `*_limited`
  metrics.
- **Multi-timestep.** The DC scoring function evaluates each timestep and combines metrics with the rule in
  `METRICS` (`optimizer/dc/genetic_functions/scoring_functions.py`): energies and penalties add, counts and
  maxima take the maximum.
- **Sign conventions.** DC fitness is $-\sum_m w_m\,\text{metric}_m$ (higher is better); a topology that islands
  the grid gets metric $+\infty$, i.e. fitness $-\infty$. AC fitness equals `overload_energy_n_1` (lower is
  better). AC acceptance (`evaluate_acceptance`) requires, against the unsplit grid, at most as many
  non-converging cases (after subtracting the number of disconnected branches), overload energy
  $\le 0.95\times$ baseline and critical branches $\le 1.1\times$ baseline; voltage-jump (> 5 %) and
  angle-difference (> 20°) counts are checked only when `enable_critical_voltage_rejection=True`. With a zero
  baseline, $1.1 \times 0 = 0$ rejects any increase.
- **Contingency propagation** (`docs/contingency_analysis/propagation.md`): ToOp expands a busbar contingency over
  all busbar sections of the same bus-breaker bus; PowSyBl's own breaker-based propagation is currently not
  enabled; pandapower groups outages by connected components.
- **Cross-tool validation.** The same N-1 in two engines:
  `packages/contingency_analysis_pkg/tests/ac_service/test_backend_comparison.py` exports IEEE 14 to MATPOWER,
  loads it in pandapower and PowSyBl and requires branch P/Q and bus $V$/$\theta$ to agree within $10^{-3}$. The
  DC oracle test `packages/dc_solver_pkg/tests/preprocessing/test_loadflows_match.py` compares the JAX solver's
  N-0 flows with pandapower `rundcpp`; its N-1 comparison is currently skipped.
- **Remedial actions** at overview level: preventive vs curative, costly (redispatch, countertrading) vs
  non-costly (topology, PSTs). Coordinated remedial action optimization (OpenRAO, CRAC, GLSK) is covered in
  [Chapter 21](../ch21-coordinated-security-rao/index.md); security-constrained OPF and optimal transmission switching
  are formulated in [Chapter 28](../../part-4-optimization/ch28-topology-optimization/index.md).

## Learning outcomes

- Compute overload energy, its power-law variant, critical branch count, double limits and `n0_n1_delta` by
  hand and in NumPy from an N-0 vector and an N-1 matrix, and reproduce ToOp's values to numerical precision.
- Run a standalone AC N-1 analysis with ToOp's contingency package and recompute its metrics from the result
  tables with pandas.
- Explain the DC and AC sign conventions and predict whether the AC stage accepts a topology, including the
  zero-baseline edge case.
- Compare an N-1 result between two power flow engines and explain the remaining differences.

## Resources

- Regulation: SO GL, Regulation (EU) 2017/1485 <https://eur-lex.europa.eu/eli/reg/2017/1485/oj/eng>:
  Art. 25 (operational security limits), Art. 32–35 (power flow limits, contingency lists, contingency
  analysis and handling), Art. 72 (operational security analysis in operational planning).
- Textbook: Wood, Wollenberg & Sheblé 3rd ed., Ch. 7 (security, contingency analysis and selection) and §8.12
  (security-constrained OPF with DC power flow). (11 syllabi) **NOT ON IA** (3rd ed.), so use a library copy.
- Courses and videos:
  - NPTEL *Economic Operations and Control of Power Systems* (IIT Kanpur, G. Mirle Vishwanath & N. P. Padhy)
    <https://nptel.ac.in/courses/108104191>: Lectures 39–41 "Power System Security I–III" (the N-1 criterion,
    the functions of security analysis, linear sensitivity factors, contingency ranking with a performance index).
  - Overbye, TAMU ECEN 460 (contingency analysis, SCOPF, LMP; slides, no videos)
    <https://overbye.engr.tamu.edu/course-2/ecen460sp25/>.
- Free: PowerWorld training modules I9 "Limit Monitoring"
  <https://www.powerworld.com/training/courses/introduction-to-powerworld-simulator/limit-monitoring> and I10
  "Contingency Analysis"
  <https://www.powerworld.com/training/courses/introduction-to-powerworld-simulator/contingency-analysis>
  (slides): how a commercial tool defines contingency lists, monitored elements and violations. The exercises
  need Simulator (Windows only). Videos (vendor training, free) from the Training playlist
  <https://www.youtube.com/playlist?list=PL0B37D8DFCE4A53A4>: "Limit Monitoring"
  <https://www.youtube.com/watch?v=BJvU7VJxoH4>, "Contingency Analysis"
  <https://www.youtube.com/watch?v=wI6Elyq8jOY>, "M4: Advanced Contingency Analysis"
  <https://www.youtube.com/watch?v=XDZkqojvfKo>, and "M5: Security-Constrained OPF"
  <https://www.youtube.com/watch?v=awoJH5-VjdE>.
- Free: PowerWorld help, "SCOPF Example: Introduction"
  <https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/SCOPF_Example_Introduction.htm>, the 7-bus
  case `B7SCOPF` that ships with Simulator. The OPF solution costs $16,011/hr, and taking line 1–2 out of service
  leaves a 32 % overload (the page words it as "line 1 to 2", which is the outaged line). With the 11
  single-line contingencies the SCOPF costs $16,048/hr and no listed outage violates a limit. The worked
  example below shows the same effect in Python, without Simulator, on a 7-bus case of our own (the
  `B7SCOPF` data is not published on the page).
- Documentation:
  - ToOp: `docs/topology_optimizer/metrics.md`, `docs/contingency_analysis/propagation.md`,
    `docs/contingency_analysis/result_filtering.md`, `docs/dc_solver/busbar_outage.md`, `docs/wall_of_shame.md`
    (known limitations), `packages/contingency_analysis_pkg/README.md` (standalone AC N-1 entry points).
  - ToOp code: fitness signs in `optimizer/dc/genetic_functions/scoring_functions.py` (DC, negated weighted sum)
    and `optimizer/ac/scoring_functions.py` (AC, `overload_energy_n_1`); double limits in `run_initial_loadflow`
    (`dc_solver/preprocess/convert_to_jax.py`); full N-1 matrices from `run_solver_symmetric`
    (`dc_solver/jax/topology_looper.py`).
  - pandapower contingency analysis <https://pandapower.readthedocs.io/en/latest/contingency.html>; pypowsybl
    security analysis <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/security.html>.
- Paper: Hedman, O'Neill, Fisher & Oren, "Optimal Transmission Switching With Contingency Analysis", *IEEE
  Trans. Power Systems* 24(3):1577–1586, 2009, doi:10.1109/TPWRS.2009.2020530.
- Optional: Billinton & Allan, *Reliability Evaluation of Power Systems* (1984). **BORROW**
  [`reliabilityevalu0000bill_l0j3`](https://archive.org/details/reliabilityevalu0000bill_l0j3).

## Worked example

Three monitored branches, one timestep, two contingencies. The worst contingency differs per branch, which is
exactly what the per-branch maximum captures.

```python
import numpy as np

f_max = np.array([100.0, 200.0, 50.0])                    # MW
n_0 = np.array([[80.0, -150.0, 49.0]])                    # (timesteps, branches)
n_1 = np.array([[[95.0, -210.0, 40.0],                    # (timesteps, contingencies, branches)
                 [120.0, -160.0, 55.0]]])

over = np.clip(np.abs(n_1) - f_max, 0, None)
overload_energy_n_1 = over.max(axis=1).sum()              # 20 + 10 + 5 = 35 MW
rel = np.clip(np.abs(n_1) / f_max, 1, None)
power_law_n_1 = ((rel**1.5 - 1) * f_max).max(axis=1).sum()     # 31.45 + 15.19 + 7.68 = 54.32
critical_n_1 = np.any(np.abs(n_1) > f_max, axis=1).sum(axis=1).max()   # 3

def double_limits(flows_tcb, f_max, lower=0.9, upper=1.0):
    flows = np.abs(flows_tcb).max(axis=(0, 1))
    low, high = flows < lower * f_max, flows > upper * f_max
    return np.where(low, lower * f_max, np.where(high, upper * f_max, flows))

print(double_limits(n_0[:, None, :], f_max))              # [ 90. 180.  49.]  (49 MW is in the dead zone)
delta = np.clip(np.abs(n_1) - np.abs(n_0)[:, None, :], 0, None).max(axis=1)   # [[40. 60.  6.]]
```

With weights `(("overload_energy_n_1", 1.0), ("critical_branch_count_n_1", 10.0))` the DC fitness of this state
is $-(35 + 10 \cdot 3) = -65$.

### The price of N-1 security on a 7-bus case, in Python

The question: how much more does it cost to run a grid so that every single line outage is survivable? Seven
buses, 11 lines (reactances in per unit on 100 MVA, limits in MW), three generators with linear costs. The LP
minimizes cost; the base-case constraints use the PTDF of [Chapter 17](../ch17-sensitivity-factors-1/index.md), and each
contingency $k$ adds $|f_\ell + \text{LODF}_{\ell k} f_k| \le f_\ell^\max$ for every other line $\ell$. Only
NumPy and SciPy are needed (`scipy.optimize.linprog` uses the HiGHS solver):

```python
import numpy as np
from scipy.optimize import linprog

# (from bus, to bus, x p.u., limit MW)
LINES = [(1, 2, .10, 240), (1, 3, .15, 240), (2, 3, .10, 200), (2, 5, .12, 240), (3, 4, .10, 200), (3, 6, .14, 200),
         (4, 5, .10, 200), (4, 6, .12, 200), (5, 6, .10, 200), (5, 7, .15, 200), (6, 7, .12, 200)]
n, m = 7, len(LINES)
load = np.array([0, 0, 80, 100, 90, 70, 0.])               # MW per bus
gens = {1: (10, 250), 2: (20, 200), 7: (35, 200)}           # bus: ($/MWh, MW max)

A = np.zeros((m, n))                                        # branch-bus incidence
for k, (i, j, *_) in enumerate(LINES):
    A[k, i - 1], A[k, j - 1] = 1, -1
b = 1 / np.array([x for _, _, x, _ in LINES])
B = A.T @ np.diag(b) @ A
ptdf = np.zeros((m, n))                                     # slack = bus 1
ptdf[:, 1:] = np.diag(b) @ A[:, 1:] @ np.linalg.inv(B[1:, 1:])
limit = np.array([f for *_, f in LINES], float)
C = np.zeros((n, len(gens)))                                # generator -> bus
for g, bus in enumerate(gens):
    C[bus - 1, g] = 1
cost = np.array([c for c, _ in gens.values()], float)

def lodf(k):                                                # flow change on every line per MW lost on line k
    h = ptdf[:, LINES[k][0] - 1] - ptdf[:, LINES[k][1] - 1]
    return h / (1 - h[k])

def flows(pg, out=None):                                    # DC flows, optionally after the outage of line `out`
    f = ptdf @ (C @ pg - load)
    if out is None:
        return f
    f = f + lodf(out) * f[out]
    f[out] = 0
    return f

def dispatch(contingencies=()):
    F, d = ptdf @ C, ptdf @ load                            # base flow = F @ pg - d
    rows, rhs = [F, -F], [limit + d, limit - d]
    for k in contingencies:
        keep = np.arange(m) != k
        Fk, dk = F + np.outer(lodf(k), F[k]), d + lodf(k) * d[k]
        rows += [Fk[keep], -Fk[keep]]
        rhs += [limit[keep] + dk[keep], limit[keep] - dk[keep]]
    return linprog(cost, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs), A_eq=np.ones((1, len(gens))),
                   b_eq=[load.sum()], bounds=[(0, p) for _, p in gens.values()], method="highs")

for name, ctg in (("OPF", ()), ("SCOPF", range(m))):
    r = dispatch(ctg)
    worst = max((np.abs(flows(r.x, k)) / limit).max() for k in range(m))
    print(f"{name:5s} cost {r.fun:6.0f} $/h  Pg {np.round(r.x, 1)}  worst post-contingency loading {100 * worst:.0f} %")
# OPF   cost   4300 $/h  Pg [250.  90.   0.]  worst post-contingency loading 104 %
# SCOPF cost   4400 $/h  Pg [240. 100.   0.]  worst post-contingency loading 100 %
```

The cheapest generator is capped at 250 MW, and the plain OPF uses all of it. The outage of line 1–2 (or 1–3)
then pushes the other one to 104 % of its limit. The SCOPF moves 10 MW from the cheap generator at bus 1 to the
one at bus 2 and pays $100/h for it: the price of N-1 security. This is the redispatch cost that ToOp tries to
avoid by switching instead of redispatching ([Chapter 28](../../part-4-optimization/ch28-topology-optimization/index.md)).

DC security is not AC security. Check the same two dispatches with pandapower's AC power flow, one outage at a
time (110 kV lines with $r = 0.1\,x$, loads with $Q = 0.3\,P$, bus 1 as the slack, which also covers the losses):

```python
import pandapower as pp

def ac_worst_loading(pg):
    net = pp.create_empty_network(sn_mva=100)
    for _ in range(n):
        pp.create_bus(net, vn_kv=110.0)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    for (bus, _), p in zip(gens.items(), pg):
        if bus != 1:
            pp.create_gen(net, bus - 1, p_mw=p, vm_pu=1.0)
    for bus, p in enumerate(load):
        if p:
            pp.create_load(net, bus, p_mw=p, q_mvar=0.3 * p)
    for i, j, x, lim in LINES:
        z = x * 110.0 ** 2 / 100                          # ohm
        pp.create_line_from_parameters(net, i - 1, j - 1, length_km=1, r_ohm_per_km=0.1 * z, x_ohm_per_km=z,
                                       c_nf_per_km=0, max_i_ka=lim / (np.sqrt(3) * 110.0))
    worst = 0
    for k in range(m):
        net.line.at[k, "in_service"] = False
        pp.runpp(net)
        worst = max(worst, net.res_line.loading_percent.drop(k).max())
        net.line.at[k, "in_service"] = True
    return worst

print(ac_worst_loading(dispatch().x), ac_worst_loading(dispatch(range(m)).x))   # about 114.7 and 109.7 %
```

The SCOPF dispatch, secure in DC, still overloads a line by about 10 % in AC (line current includes the reactive
flow and the losses, which the DC model drops). PowerWorld's SCOPF closes this gap by iterating between the LP
and the power flow equations; ToOp does the same in two stages, a fast DC search followed by AC validation
([Chapter 16](../ch16-dc-power-flow/index.md)).

## Documentation vs code: known discrepancies

- `docs/topology_optimizer/metrics.md` calls `exponential_overload_energy` "exponentially weighted"; the code is
  the power law above.
- The same page says N-1 overload energy "considers the worst contingency for each timestep"; the code takes the
  worst contingency **per branch** and sums over branches and timesteps.
- `choose_max_mw_flow` (`dc_solver/jax/aggregate_results.py`) lists `critical_branch_count_limited_n_1` in its
  N-0 group: `critical_branch_count_limited_n_1` is evaluated against the N-0 limited limits, and
  `critical_branch_count_limited_n_0` falls through to the unlimited `max_mw_flow`.
- The `double_limit_n0`/`double_limit_n1` docstrings in `PreprocessParameters`
  (`interfaces/messages/preprocess/preprocess_commands.py`) describe only the 90 % rule; the
  dead zone between 90 % and 100 % is documented only in `compute_double_limits`.
- `metrics.md` says `bb_outage_penalty` is 0 when the topology is equal or better under busbar outages; that holds
  only with `clip_bb_outage_penalty=True` (default `False`).
- DC `critical_branch_count_n_1` counts the worst timestep; the AC version (`count_critical_branches`) counts
  distinct `(timestep, element)` pairs. They agree for one timestep.

## Lab

!!! example "Lab: recompute ToOp's security metrics by hand, in DC and AC"

    Tools: ToOp container ([Chapter 0](../../part-0-orientation/ch00-orientation-and-setup/index.md)), Jupyter with NumPy,
    pandas and matplotlib ([Chapter 30](../../part-5-computing/ch30-scientific-python/index.md)), pypowsybl.

    !!! warning "Do not use `notebooks/run_ac_contingency_analysis.ipynb`"

        That notebook does not run: it passes `GridElement` where `Nminus1Definition.monitored_elements`
        requires `MonitoredElement` (Pydantic raises `ValidationError`), its pandapower cell has a syntax error
        (`method="ac" n_processes=15` is missing a comma) and calls `run_contingency_analysis_pandapower` with
        arguments it no longer accepts (it now takes `cfg`). CI never executes it because
        `notebooks/tests/test_notebooks.py` only collects `example*.ipynb`. Build the N-1 definition
        yourself instead, as in Part B below.

    **Part A: DC metrics from the full N-1 matrix.**

    1. As in `notebooks/example1_dc_loadflow_example.ipynb`, call `case57_data_powsybl(folder)` and
       `load_grid(...)`. Record `stats` and the unsplit `(overload_energy_n_0, overload_energy_n_1)` returned by
       `run_initial_loadflow` (`dc_solver/preprocess/convert_to_jax.py`).
    2. Build a batch of unsplit topologies with `default_topology(solver_config, batch_size=4)`
       (`dc_solver/jax/topology_computations.py`) and obtain the full matrices with
       `run_solver_symmetric(topologies, None, None, dynamic_information, solver_config, fn)`
       (`dc_solver/jax/topology_looper.py`), where `fn` is a pass-through aggregation function such as
       `lambda lf: (lf.n_0_matrix, lf.n_1_matrix)` defined **once**, outside any loop.
    3. In NumPy, recompute both overload energies for the unsplit topology from `n_0`, `n_1` and
       `dynamic_information.branch_limits` (`max_mw_flow`, `max_mw_flow_n_1` if not `None`, and
       `overload_weight`: the case57 generator writes line weights of 2.0). Match ToOp to at least six digits.
    4. Recompute `max_mw_flow_limited` and `max_mw_flow_n_1_limited` with your own double-limit function and
       compare them with the arrays stored in `branch_limits`. Plot, with matplotlib, relative loading against
       the ratio limited/rating to show the three regimes.
    5. Put real action indices into the first slot of two topologies (`topologies.action.at[1, 0].set(...)`,
       rebuilt as `ActionIndexComputations(action=..., pad_mask=topologies.pad_mask)`) and compute the power-law
       overload energy, critical branch count and `n0_n1_delta` for all of them. Check each value by passing
       `functools.partial(aggregate_to_metric, branch_limits=..., reassignment_distance=..., n_relevant_subs=...,
       metric=...)` (`dc_solver/jax/aggregate_results.py`) as the aggregation function; build each partial once.

    **Part B: AC N-1 and metrics from result tables.**

    1. Load `data/grid_node_breaker/grid.xiidm` with `pypowsybl.network.load` and build an `Nminus1Definition`
       (`interfaces/nminus1_definition.py`): `monitored_elements` holds one `MonitoredElement(id=..., type="LINE",
       kind="branch")` per line (and per transformer; `kind="bus"` for buses), `contingencies` holds one
       `Contingency(id=..., elements=[GridElement(id=..., type="LINE", kind="branch")])` per outaged branch, plus
       a base case `Contingency(id="BASECASE", elements=[])`. `packages/contingency_analysis_pkg/README.md`
       lists the entry points.
    2. Run `get_ac_loadflow_results(net, nminus1_def, job_id="lab", timestep=0, n_processes=1)` and collect
       `branch_results`, `node_results` and `va_diff_results` into pandas.
    3. Recompute `overload_energy_n_1` (field `p`), `overload_current_n_1` (field `i`),
       `critical_branch_count_n_1` and `max_va_diff_n_1` with pandas group-bys, then compare with
       `compute_metrics(lf_results, base_case_id="BASECASE")` from `contingency/ac_loadflow_service/compute_metrics.py`.
    4. Explain why the number of branch result rows is below
       $(2 \cdot n_\text{lines} + 2 \cdot n_\text{trafos} + 3 \cdot n_\text{trafo3w}) \cdot n_\text{contingencies}$.
    5. Optional cross-tool check: build IEEE 14 in pandapower, export it with `pandapower.converter.to_mpc`,
       load the `.mat` file in pypowsybl, run the same single-branch N-1 in both (pandapower ids need
       `id_type="unique_pandapower"`) and plot the distribution of branch-flow differences, as
       `packages/contingency_analysis_pkg/tests/ac_service/test_backend_comparison.py` does.

    **Part C (optional): the price of security on the 7-bus case.** The two code blocks of the worked example are
    also a module, [`case7.py`](src/case7.py), whose `main()` prints the answers to
    steps 1–3.

    1. Run the two code blocks of "The price of N-1 security on a 7-bus case, in Python" and print, for the SCOPF
       dispatch, which contingency constraints are binding (post-contingency loading of exactly 100 %).
    2. Make the AC check pass, as PowerWorld's SCOPF does by iterating: for each line whose worst AC loading is
       above 100 %, set its limit to (worst DC post-contingency flow) × 100 / (worst AC loading), solve the SCOPF
       again and repeat until the AC worst loading is at most 100 %. Report the extra cost of AC security over DC
       security (about $4,400 → $4,670/h after two updates).
    3. Lower the limit of line 1–3 to 150 MW. The OPF cost stays at $4,300/h, but the SCOPF cost rises to about
       $5,360/h. Explain why the gap grows, and why an operator would look for a remedial action such as a
       topology change before paying it ([Chapter 28](../../part-4-optimization/ch28-topology-optimization/index.md)).
    4. If you have PowerWorld Simulator, enter the same 7 buses and 11 lines, solve OPF and SCOPF, and compare
       dispatch and cost with your result.
    5. **Hand the case to ToOp and ask the topology question instead.** ToOp has no OPF (no costs, no
       redispatch); what it can do is look for a bus split that removes the N-1 overload at zero redispatch
       cost. Exercises 7–10 do this step by step, with solutions; their script is
       [`case7_toop.py`](src/case7_toop.py).

## Exercises

1. Pen and paper: for the worked example, find a flow change on one branch in one contingency that leaves
   overload energy unchanged but increases the critical branch count, and one that does the opposite. What does
   the optimizer prefer under each objective?
2. *(Jupyter: NumPy + matplotlib)* Plot the per-branch penalty of overload energy, the $\alpha = 1.5$ power law
   and cumulative overload as a function of loading from 0 to 200 % for limits of 100 MW and 1 000 MW. Which
   metric makes a 10 % overload on a 110 kV line as costly as on a 380 kV line?
3. *(Jupyter: NumPy)* Show with a two-branch example that the double-limit dead zone can make the unsplit grid
   have zero `overload_energy_limited_n_0` while a topology that only redistributes flow between the two
   branches gets a positive value. Relate this to why the rule exists.
4. Pen and paper: an unsplit grid has AC `overload_energy_n_1` = 0, `critical_branch_count_n_1` = 0 and 2
   non-converging cases. A candidate has 0 MW, 1 critical branch and 2 non-converging cases after disconnecting
   one branch. Apply `evaluate_acceptance` with default thresholds and state the rejection criterion, if any.
   Would a relative tolerance or an absolute slack be the better design?
5. *(Octave + MATPOWER)* For `case30`, compute the DC N-1 flow matrix with `makePTDF` and `makeLODF`
   (<https://matpower.org/docs/ref/matpower7.1/lib/makeLODF.html>), skipping bridges, set every limit to 1.2 times
   the base-case flow, and compute overload energy and critical branch count. Repeat after doubling one load
   and explain which contingencies became worst for which branches.
6. *(Jupyter: pandas)* The DC optimizer's fitness for a run is `-713` and the AC fitness of the same topology is
   `12.4`. Write down the weighted objective that could have produced the DC value and explain why the two
   numbers cannot be compared directly.

Exercises 7–10 continue the 7-bus case of the worked example inside ToOp. Their scripts are in the chapter's
`src` folder: [`case7.py`](src/case7.py) (SciPy and pandapower, runs anywhere) and
[`case7_toop.py`](src/case7_toop.py) (runs in the ToOp container). Solve each exercise
before opening its solution; the numbers quoted were produced with ToOp `ba1874e` on CPU.

7. *(ToOp container)* **Same N-1, two tools.** Wrap the 7-bus grid with the OPF dispatch of the worked example
   for ToOp, as `case14_pandapower` does in `dc_solver/example_grids.py`: bus 1 as `ext_grid` (the slack),
   `grid.json` plus the masks, buses 3–6 as relevant substations, line outages only. Load it with `load_grid` and
   `run_initial_loadflow` (`dc_solver/preprocess/convert_to_jax.py`), run the unsplit topology through
   `run_solver_symmetric` with a pass-through aggregation function and read the full N-1 matrix. Compare
   `overload_energy_n_1` and the worst post-contingency loading with the SciPy code, and derive the value of
   `overload_energy_n_1` from the definition of the metric.

    ??? note "Solution"

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

8. *(Pen and paper, then ToOp)* **Count the action set.** Before loading the case, predict how many actions ToOp
   builds for the relevant substations 3–6 when a split must leave at least two branches on each busbar, mirror
   images (the two busbars swapped) count once, and the unsplit configuration is one action per substation.
   Derive a formula in the number of branches $d$ of a substation, then check it against
   `dynamic_information.action_set.n_actions_per_sub` and decode the actions with `branch_actions`,
   `substation_correspondence` and `network_data.branches_at_nodes`. Which substation is missing, and why?

    ??? note "Solution"

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

9. *(Pen and paper, then ToOp)* **An overload no switch can fix.** Prove that, after the outage of line 1–2, the
   flow on line 1–3 is 250 MW for every topology of buses 3–6. Then run the twelve actions of Exercise 8 through
   `run_solver_symmetric` in one batch (`ActionIndexComputations` with one action in the first slot of each
   topology) and report, per action, the worst post-contingency loading and the largest change of a base-case
   flow. Which actions make things worse, which one islands the grid, and what is the only remedy?

    ??? note "Solution"

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

10. *(SciPy, ToOp and pandapower)* **An overload a switch can fix.** Start from the SCOPF dispatch (240, 100 and
    0 MW) and lower the limit of line 3–6 to 100 MW. (a) With the SciPy code, find the new worst post-contingency
    loading of that dispatch and the cost of the SCOPF that respects the new limit. (b) Rebuild the ToOp folder
    with this dispatch and limit (`report(folder, pg, limit)` in the script does both) and find which single
    splits clear every N-1 overload. (c) Reproduce the best split in pandapower: add a bus, move the lines of the
    split to it (`split_bus` in `case7.py`) and run the DC N-1 again. (d) Explain, from the paths available after
    the L2-5 outage, why this overload is fixable while the one of Exercise 9 is not, and check the split with
    the AC N-1 as well.

    ??? note "Solution"

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
