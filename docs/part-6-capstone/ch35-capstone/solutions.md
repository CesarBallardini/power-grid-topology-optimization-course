# Chapter 35 – Solutions

Solutions of the Lab and the Exercises of [Chapter 35](index.md) that have one; the projects and the operator brief
are team work and have no single answer, but the Lab below is the baseline every team starts from. The code is in
the chapter's `src` folder. Four programs read ToOp's output files and run anywhere: [`res_table.py`](src/res_table.py),
[`switching_distances.py`](src/switching_distances.py), [`brief.py`](src/brief.py) and
[`hvdc_sensitivity.py`](src/hvdc_sensitivity.py). Three need ToOp's interpreter: [`baseline_toop.py`](src/baseline_toop.py)
(the staged pipeline, three runs), [`ac_check_toop.py`](src/ac_check_toop.py) and [`sld_toop.py`](src/sld_toop.py);
[`test_ac_check_toop.py`](src/test_ac_check_toop.py) holds the two-line reproduction of Exercise 2. The files under
[`src/baseline/`](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-6-capstone/ch35-capstone/src/baseline){ target="_blank" }
(`res.json`, the two configurations, `unsplit_ac_metrics.json`, and `ac_metrics.json` plus `orao_summary.json` per
validated topology) are what `baseline_toop.py` wrote with ToOp `ba1874e` in the course's ToOp container (image
`toop-course:cpu`, `docker run --cpus 4`, the work folders in the named volume `toop-course-runs`): the node-breaker
example grid three times with seed 1 and 10 s of runtime (`python baseline_toop.py <work> <grid.xiidm> 3 10`), and the
complex example grid (`data/complex_grid/grid.xiidm`, an HVDC link; its preprocessing log reports 67 branches, 9
relevant substations and 121 actions) once with 25 s (`... 1 25`), because the node-breaker grid has one relevant
substation and ten actions and cannot show more than two repertoire cells. An earlier run of the same script on the
maintainer's Windows host, with the shims below, gave the same DC and AC figures; only the number of epochs differed.
Every number below comes from these files or from the programs (`python <name>.py`; the `_toop` ones in the
container).

Two Windows notes carried over from [Chapter 34](../../part-5-computing/ch34-pipeline-engineering/solutions.md): the
colons in ToOp's TensorBoard folder names (`baseline_toop.windows_shim`) and the zero time step between fast epochs
(`iterations_per_epoch=500`); both are harmless in the container. Two more came up here: `PreprocessParameters(preprocess_bb_outages=True)` fails on a
bus-breaker model such as `data/case57_xiidm/grid.xiidm` (`convert_rel_bb_outage_data` meets
`rel_bb_outage_zero_flow_br_indices = None`), and the importer's default area settings leave that case57 file with no
branches at all, so the capstone's richer grid is the complex example grid.

## Lab

1. **Baseline run.** `baseline_toop.baseline(work_dir, runs=3, seed=1, runtime_seconds=10)` copies `grid.xiidm`
   into the work folder, runs `run_preprocessing` once (`PreprocessParameters(action_set_clip=2**10,
   preprocess_bb_outages=True)`, 8 s) and then, per run, `run_dc_optimization_stage` with
   `BatchedMEParameters(random_seed=1, runtime_seconds=10, iterations_per_epoch=500, n_worst_contingencies=2,
   me_descriptors=(split_subs, 4 cells))`, `LoadflowSolverParameters(max_num_splits=3, batch_size=8)` and
   `perform_ac_analysis` with `k_best_topos=4`; `dc_config.json` and `ac_config.json` are written next to the
   outputs, which is what goes into the team repository. `baseline_toop.tabulate`:

    | Run | `initial_fitness` | `max_fitness` | Iterations | `best_topos` | Validated | Accepted |
    |---|---|---|---|---|---|---|
    | run_0 | −61.445 | −42.345 | 39 | 1 | 1 | 0 |
    | run_1 | −61.445 | −42.345 | 55 | 1 | 1 | 0 |
    | run_2 | −61.445 | −42.345 | 55 | 1 | 1 | 0 |

    The three runs end with the same best topology (action 5, DC overload energy 42.345 MW against 61.445 MW
    unsplit) but with 39, 55 and 55 epochs: a fixed seed makes the random stream reproducible, a fixed
    wall-clock budget does not make the amount of work reproducible, which is why Chapter 26 asks for budgets in
    evaluations. The AC validation rejects the topology every time: `evaluate_acceptance` returns the criterion
    `overload-energy` with `value_after = 25.377 MW`, `value_before = 16.336 MW` (the unsplit grid in AC) and the
    threshold 0.95, so a topology that removes 19 MW of DC overload energy adds 9 MW of AC overload energy. Noted
    for the brief: the DC and AC figures of the same unsplit grid differ by a factor of almost four (61.4 against
    16.3 MW), and 7 of the 18 AC contingencies come back without a result (status `NO_CALCULATION`, which ToOp
    counts as non-converging; step 3 shows which and why).

2. **Trace.** Action 5 of `action_set.json` is the fifth `local_actions` entry: a split of the station `VLevel2`
   (voltage level `VL2`) that moves line `L12` to the second busbar, realized by opening the coupler `VL2_BREAKER`,
   opening the disconnector `L12_DISCONNECTOR_4_0` and closing `L12_DISCONNECTOR_4_2` (the `orao_summary.json` of
   topology 0; `disconnections` is empty). In the DC stage the genotype of a candidate is a vector of action indices
   per relevant substation (`optimizer/dc/genetic_functions/genotype.py`, mutated in `mutation/`); for a batch of
   candidates `compute_symmetric_batch` (`dc_solver/jax/compute_batch.py`) calls `compute_bsdf_lodf_static_flows`,
   which applies the BSDF update of the PTDF for each split (`bsdf.py`), the MODF update for the disconnections
   (`multi_outages.py`), then the LODF N-1 flows; `aggregate_results.py` turns the N-1 matrix into
   `overload_energy_n_1` (42.345 MW), `max_flow_n_1` (2.207), `split_subs` (1), `switching_distance` (2: the
   precomputed `reassignment_distance` of action 5) and `top_k_overloads_n_1` (67.27 MW with `n_worst_contingencies
   = 2`). The repertoire (`optimizer/dc/repertoire/discrete_me_repertoire.py`) has one cell per value of the
   descriptor `split_subs`; the candidate lands in cell 1, beats whatever was there, and `best_topos` lists the best
   of every non-empty cell, here just this one, since the unsplit grid is the initial topology and not a repertoire
   member. The AC worker's `select_strategy` (`optimizer/ac/select_strategy.py`) filters the DC pushes stored in
   its database with the median, dominator and discriminator masks over the target metrics, and
   `compute_metrics_single_timestep` (`optimizer/ac/scoring_functions.py`) scores the realized topology from the AC
   N-1 results (`fitness = overload_energy_n_1 = 25.377`). `res_table.weighted_sum_check` confirms the DC fitness:
   `target_metrics = {overload_energy_n_1: 1.0}`, so `fitness = −1.0 × 42.3453 = −42.3453`, and the check holds for
   every row of every baseline file.

3. **Independent AC check.** `ac_check_toop.compare` rebuilds the AC result of topology 0 with
   `create_loadflow_runner(data_folder, grid_xiidm).run_ac_loadflow([5], [])` followed by
   `compute_metrics_single_timestep`: fitness 25.37695571260243 against 25.37695571260243 in `ac_metrics.json`,
   the largest gap over the 17 shared metrics being 0.0 (on the complex grid, $9 \times 10^{-13}$): this route is
   ToOp's own code path and agrees to the last bit, so it checks the files, not the method. `independent_n0`
   loads `topology_0/modified_network.xiidm` with pypowsybl alone, runs an AC load flow and divides each line's
   current by its permanent limit from `get_operational_limits()`: 0.84607 on `L3`, against `max_flow_n_0 =
   0.84607` in the file; for the unsplit grid 0.8652 on `L1` against 0.8652. On the complex grid the same
   lines-only figure is 0.979 (unsplit) and 1.704 (topology 0) against 1.241 and 1.945 in the files, because
   there the worst monitored element is a transformer, which the lines-only check leaves out: an independent check
   is only as good as its coverage of the monitored set. The lab's other route, `ac_check_toop.n1_route`, runs
   `get_ac_loadflow_results` outside the runner on `topology_N/modified_network.xiidm` with the preprocessed
   `nminus1_definition.json` (its monitored elements are `MonitoredElement` objects: 8 on the node-breaker grid,
   with 19 contingencies including the base case; 95 and 128 on the complex grid) and sums the overload energy
   again in pandas. With pypowsybl's default load-flow parameters it does **not** reproduce the files: the unsplit
   node-breaker grid gives 112.25 MW, 3 critical branches and no failed contingency, against 16.34 MW, 2 and 7 in
   `unsplit_ac_metrics.json`. With the parameters the preprocessing stored (`loadflow_parameters.json`, which
   `create_loadflow_runner` loads; `ac_check_toop.stored_lf_params`) it matches every validated topology on both
   grids: 25.37695571260243 MW for node-breaker topology 0, 1669.53 MW, 29 critical branches and 5 failed
   contingencies for complex topology 0, and the unsplit figures exactly. The 7 failures of the node-breaker grid
   are the three injection outages (`generator3`, `load1`, `load2`) and four of the seven busbar outages, all with
   status `NO_CALCULATION`: they are not divergences, they are cases the load flow refused to compute under the
   stored parameters, and with the default ones the same contingencies converge and add 96 MW of overload energy.
   On the complex grid the default parameters give 4 `MAX_ITERATION_REACHED` cases (one more line, `L1`) and
   2166.60 MW against 2160.39 MW. An independent check therefore has to fix the load-flow parameters as well as the
   N-1 definition, and the brief should say that 7 of 18 contingencies were not computed, rather than "did not
   converge".

4. **Draft brief.** `sld_toop.diagrams(grid.xiidm, topology_0/modified_network.xiidm, orao_summary)` finds the
   voltage level of the three switches (`VL2`), runs an AC load flow on both networks and renders both with
   `get_single_line_diagram_custom(..., highlight_grid_model_ids=[the three switch ids])`: two SVG files of 45,735
   and 46,475 characters with 3 highlighted elements each, one per switch (ToOp adds the class `highlight` to the
   matching elements; the colour `#0050fcff` itself occurs 7 times in every diagram because it sits in the CSS, once
   per voltage class, so counting the colour says nothing), and the switch states read from both networks, `VL2_BREAKER` closed → open,
   `L12_DISCONNECTOR_4_0` closed → open, `L12_DISCONNECTOR_4_2` open → closed. `brief.brief(run_dir)` writes the
   draft: situation (unsplit AC N-1: worst N-0 loading 0.865, worst N-1 loading 1.235, overload energy 16.34 MW, 2
   critical branches, 7 non-converging contingencies), one proposal (topology 0, one split, AC overload energy
   25.38 MW, 1 critical branch, 7 non-converging contingencies, largest angle difference 1.8°, ratio 1.55 to the
   unsplit grid: not recommended), the switching steps in operator language with a feasibility note each (open the
   breaker `VL2_BREAKER`, a breaker can interrupt load current; open the disconnector `L12_DISCONNECTOR_4_0` and
   close `L12_DISCONNECTOR_4_2`, disconnectors carry no load current, so the breaker in series must already be
   open), and the assumptions (the N-1 definition, the unexplained non-convergences, the diagrams). The honest
   brief for this grid says: no topology to propose, redispatch or accept the overload; the reviewing team should
   ask why 7 of the 18 AC contingencies are not computed (step 3) before anything else.

5. **Pick your discrepancy.** Two reproductions are in the folder. (a) The silent drop of
   `worst_k_contingent_cases`: `Metrics(fitness=0.0, extra_scores={}, worst_k_contingent_cases=['L8']).worst_k_contingency_cases`
   is `None` (two lines), and the test `test_ac_metrics_carry_the_worst_contingencies` in
   `test_ac_check_toop.py`, which rebuilds the AC metrics of topology 0 and asserts that
   `worst_k_contingency_cases` is not `None`, fails on `ba1874e` (it is marked `xfail(strict=True)` so that the
   course's suite stays green and flips when ToOp fixes it): every `ac_metrics.json` of the baseline indeed has
   `"worst_k_contingency_cases": null`. (b) Not on the chapter's list: the tracked preprocessed folder
   `data/grid_node_breaker/grid` of `ba1874e` was written by an older ToOp and its `action_set.json` carries
   `starting_topology` and `simplified_starting_topology` instead of `starting_bus_groups` and
   `simplified_starting_bus_groups`; `ActionSet` forbids extras, so `create_loadflow_runner` raises a
   `ValidationError` with four errors and the AC stage of `perform_ac_analysis` cannot run on the tracked folder,
   which is why `baseline_toop` preprocesses the grid again rather than reusing it (Chapter 33, Exercise 3 shows
   the other side of `extra='forbid'`: it is the strict model that caught the stale fixture).

## Exercise 1

`res_table.best_topos_table` flattens `best_topos` into one row per repertoire cell with `actions`,
`disconnections`, `fitness` and every `extra_scores` key, and `weighted_sum_check` recomputes
$\mathrm{fitness} = -\sum_m w_m\, \mathrm{extra\_scores}_m$ over `args.ga_config.target_metrics`. Node-breaker
grid, every run: one row, actions `[5]`, fitness −42.345326, `max_flow_n_1` 2.207, `overload_energy_n_1` 42.345,
`split_subs` 1, `switching_distance` 2, `top_k_overloads_n_1` 67.271; recomputed −42.345326, match. Complex grid
(`target_metrics = {overload_energy_n_1: 1.0}`, initial fitness −1310.860, four cells):

| Actions | Fitness | `max_flow_n_1` | `overload_energy_n_1` | `split_subs` | `switching_distance` | `top_k_overloads_n_1` |
|---|---|---|---|---|---|---|
| `[1, 54, 68]` | −1160.366 | 2.663 | 1160.366 | 3 | 8 | 1104.256 |
| `[54, 68]` | −1169.118 | 2.728 | 1169.118 | 2 | 4 | 665.022 |
| `[69]` | −1184.119 | 2.663 | 1184.119 | 1 | 4 | 721.598 |
| `[]` | −1243.819 | 2.663 | 1243.819 | 0 | 0 | 698.462 |

`fitness` equals the negated weighted sum in every row of every file (`matches` is True throughout; with one
target metric of weight 1 the check is $\mathrm{fitness} = -\mathrm{overload\_energy\_n\_1}$, which is also why
`max_fitness` in `res.json` is the first row). The operator who accepts at most one split substation gets the best
row with `split_subs ≤ 1`, `for_operator(table, 1)`: action `[69]`, fitness −1184.119, which removes 60 MW of DC
overload energy (5 %) against 83 MW for three splits, with the same `max_flow_n_1`; on the node-breaker grid the
only candidate, `[5]`, qualifies. And the pick comes with its AC verdict from the baseline: `[69]` is rejected
(`overload-energy`, 2154.997 against 0.95 × 2160.388 MW), as are the other three (two of them by `convergence`, 5
non-converging contingencies against 3 unsplit), so the honest recommendation is none of them.

```python title="res_table.py"
--8<-- "docs/part-6-capstone/ch35-capstone/src/res_table.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-6-capstone/ch35-capstone/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

The two-line reproduction (`test_metrics_keyword_is_silently_dropped_in_two_lines` in `test_ac_check_toop.py`):

```python
metrics = Metrics(fitness=0.0, extra_scores={}, worst_k_contingent_cases=['L8'])
assert metrics.worst_k_contingency_cases is None
```

`Metrics` (`optimizer/interfaces/messages/results.py`) keeps Pydantic's default `extra='ignore'`, so the
misspelled keyword that `compute_metrics_single_timestep` passes (`optimizer/ac/scoring_functions.py`: `metrics.pop("worst_k_contingent_cases", None)` and
`Metrics(..., worst_k_contingent_cases=...)`) is accepted and discarded, and every AC `Metrics` leaves the worker
with `worst_k_contingency_cases = None`; downstream, `get_early_stopping_contingency_ids` sees an empty list and
logs "No overload threshold or case ids found", so the AC early stopping never gets the DC worst cases. The
failing test is the second one in the same file, written against ToOp's own API so that it can move into
`packages/topology_optimizer_pkg/tests/`: run the AC load flow of a topology through `create_loadflow_runner`,
call `compute_metrics_single_timestep`, assert `worst_k_contingency_cases is not None`. It fails on `ba1874e`
(marked `xfail(strict=True)` here) and passes once the keyword is spelled `worst_k_contingency_cases` on both the
`pop` and the constructor. Pull request, as `.github/workflows/validate_pr_title.yaml` (commitizen's conventional
commits) accepts it:

> **Title:** `fix: pass worst_k_contingency_cases to Metrics in the AC scoring`
>
> **Body:** `compute_metrics_single_timestep` builds `Metrics` with the keyword `worst_k_contingent_cases`, which
> the model does not declare; Pydantic ignores it, so the AC metrics never carry the worst contingencies and the
> early-stopping subset falls back to the full N-1 list (`get_early_stopping_contingency_ids` logs a warning on
> every topology). This renames the keyword in the `pop` and in the constructor, adds
> `test_ac_metrics_carry_worst_contingencies` (fails before, passes after), and sets `extra="forbid"` on `Metrics`
> so that the next misspelling is a `ValidationError`. Fixes #<issue>. Signed-off-by: <name> <email>.

The commit is signed off (`git commit -s`, the DCO check), the branch comes from a fork, and the hooks run on
the host (`pre-commit install`); the `extra="forbid"` line is the part a reviewer may push back on, because other
producers of `Metrics` may rely on the leniency, which is exactly the discussion the pull request should start.

## Exercise 3

`sld_toop.diagrams` and `brief.switching_steps` on topology 0 of the node-breaker baseline (the station `VL2`):
the before and after diagrams are rendered from `grid.xiidm` and `topology_0/modified_network.xiidm` after an AC
load flow, with `highlight_grid_model_ids` set to the three switch ids of `orao_summary.json`, which the custom
style paints in `#0050fcff` (3 highlighted elements in each SVG, one per switch, counted by their `highlight`
class).
Read side by side, the before diagram shows the two busbars of `VL2` coupled through the closed `VL2_BREAKER`,
with line `L12` on the first busbar (`L12_DISCONNECTOR_4_0` closed, `_4_2` open); the after diagram shows the
coupler open and `L12` on the second busbar. The switching steps in operator language:

1. Open the busbar coupler breaker `VL2_BREAKER` (a breaker: it can interrupt load current; afterwards the two
   busbars of `VL2` run separately, with the feeders as they are).
2. Open the disconnector `L12_DISCONNECTOR_4_0` to take line `L12` off busbar 1 (a disconnector: it must not carry
   current, so `L12` has to be de-energised or its breaker open at this moment; ToOp's switch list says nothing
   about the line breaker, which is the first thing the operator adds).
3. Close the disconnector `L12_DISCONNECTOR_4_2` to connect `L12` to busbar 2 (again without current; then
   re-close the line breaker, and if `L12` was kept live from the far end, synchro-check: the two ends are the
   same synchronous grid, the angle difference is the one across the open path, small here, 1.8° at most in N-1).

ToOp's ORAO summary gives the end state of the switches, not an operable sequence: the order above (coupler
first, disconnectors with the feeder dead) is the operator's, and the `orao_summary.json` of `ba1874e` emits only
`SWITCH` entries (no `TERMINALS_CONNECTION`, no PST setpoints), as the chapter's discrepancy list notes. For the
complex grid the same program produces one pair of diagrams per split station (`VL_2W_MV_HV_MV`, `VL_MV`,
`VL_3W_HV` for the three-split topology, 52 000 to 60 000 characters each).

```python title="sld_toop.py"
--8<-- "docs/part-6-capstone/ch35-capstone/src/sld_toop.py"
```

```python title="brief.py"
--8<-- "docs/part-6-capstone/ch35-capstone/src/brief.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-6-capstone/ch35-capstone/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`switching_distances.all_pairs` matches each `topology_*/ac_metrics.json` with its entry of `res.json`:

| Grid | Actions | DC `switching_distance` | AC `switching_distance` | DC overload energy | AC overload energy |
|---|---|---|---|---|---|
| node-breaker (3 runs) | `[5]` | 2 | 3 | 42.3 | 25.4 |
| complex | `[1, 54, 68]` | 8 | 12 | 1160.4 | 1669.5 |
| complex | `[54, 68]` | 4 | 7 | 1169.1 | 1683.1 |
| complex | `[69]` | 4 | 5 | 1184.1 | 2155.0 |
| complex | `[]` | 0 | (not reported) | 1243.8 | 2160.4 |

The plot (`switching_distances.plot`, DC on the horizontal axis, AC on the vertical, one point per topology and
the diagonal for reference) has every point above the diagonal: the AC value exceeds the DC value by 1 to 4 and
the two are equal for none of the seven topologies, and the gap grows with the number of splits. The two numbers
are not the same quantity. The DC stage (`get_switching_distance` in `dc_solver/jax/aggregate_results.py`) sums,
over the applied branch actions, a `reassignment_distance` computed once during preprocessing, the number of
branch reassignments of the action against the starting configuration; the AC stage (`extract_switching_distance`
in `optimizer/ac/scoring_functions.py`) counts the entries of the realized topology's `branch_reassignment_diff`
plus `injection_reassignment_diff` after the action set has been applied to the real node-breaker model, where
moving a line also moves what hangs on it and a coupler counts as an operation. That is the note of
`docs/topology_optimizer/metrics.md` ("the AC solver counts this slightly differently than the DC solver. This is
a bug, not a feature"): the same topology has two switching distances, the DC one is a repertoire descriptor
(`me_descriptors` can bin on it) and the AC one is what the brief should quote, since it is closer to the
operations of Exercise 3; and the unsplit topology has no AC value at all, because the AC metrics are only
computed for `k_best_topos` candidates, the unsplit grid among them only by accident (the complex grid's cell 0).

```python title="switching_distances.py"
--8<-- "docs/part-6-capstone/ch35-capstone/src/switching_distances.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-6-capstone/ch35-capstone/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

In the DC model the AC grid does not see the HVDC link as a branch: the converter at bus $i$ takes $P$ out of the
AC grid and the converter at bus $j$ puts $P$ back (losses neglected), so the link is a pair of injections, $-P$
at $i$ and $+P$ at $j$, and the AC branch flows are $f = \mathrm{PTDF}\,(p - P e_i + P e_j)$. The PTDF is linear
in the injections, so a setpoint change $\Delta P$ changes the flows by

$$
\Delta f = \mathrm{PTDF}\,(-\Delta P\, e_i + \Delta P\, e_j) = \Delta P\,(\mathrm{PTDF}_{\cdot,j} - \mathrm{PTDF}_{\cdot,i}),
$$

which is the chapter's $\Delta P\,(\mathrm{PTDF}_{\cdot,i} - \mathrm{PTDF}_{\cdot,j})$ with the opposite
convention for the direction of the link (positive setpoint from $j$ to $i$). If $i$ is the slack, its column is
zero and the sensitivity is $\Delta P\,\mathrm{PTDF}_{\cdot,j}$: the setpoint then moves power between the slack
and $j$, which is consistent, because the slack absorbs the balance. `hvdc_sensitivity.check` re-solves a
six-bus meshed grid with the two injections changed by $\Delta P = 0.37$ and compares with the column: the
largest difference over all 30 ordered pairs is $1.5 \times 10^{-16}$; the column of a link 2 → 5 is
`[0.3753, 0.5899, -0.4101, -0.2735, -0.488, 0.512, 0.1366, -0.2146]` MW per MW, and the net flow leaving bus 2
changes by exactly −1 MW per MW of setpoint (Kirchhoff at the converter bus). The column is the same object as a
PSDF column of Chapter 17 with the branch-side term left out: a PST is a loop-flow source on its own branch, an
HVDC link an injection pair with no AC branch in between.

Where it would live in ToOp: ToOp already treats controllable PSTs as "injections that are angles",
`NodalInjectionInformation` (`dc_solver/jax/types.py`) with `controllable_pst_indices` (nodes whose injection the
optimizer may vary), `pst_tap_values` and `starting_tap_idx`, written into the nodal injection tensor by
`write_pst_taps_to_nodal_injections` (`dc_solver/jax/pst.py`) and optimized by `nodal_inj_optim.py`; the N-0
flows are then recomputed as $\mathrm{PTDF}\cdot p$ by `update_n0_for_pst_taps`. An HVDC setpoint fits the same
slot with two nodes instead of one: an `hvdc_from_node`, `hvdc_to_node` pair with a setpoint range
(`HVDCRange` in `interfaces/stored_action_set.py`, which `network_data.py` still writes as `hvdc_ranges=[]`), the
setpoint carried as traced data next to `pst_tap_idx` in the genotype, and the column $\mathrm{PTDF}_{\cdot,j} -
\mathrm{PTDF}_{\cdot,i}$ never stored, because it is read off the PTDF of the current topology at run time, after
the BSDF updates, exactly as the PST path does. The preprocessing side would gather the converter stations the
PowSyBl backend already collects in `_get_injections` (`dc_solver/preprocess/powsybl/powsybl_backend.py`) and
their `pmax`, which is the project 7 plan of the chapter.

```python title="hvdc_sensitivity.py"
--8<-- "docs/part-6-capstone/ch35-capstone/src/hvdc_sensitivity.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-6-capstone/ch35-capstone/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 6

One possible plan, for project 1 (clean-room solver), sized for a team and a semester; any project needs the same
seven headings.

- **Research question.** Can the published papers alone (arXiv:2501.17529 for PTDF/LODF, the BSDF paper, the MODF
  paper arXiv:1606.07276) reproduce ToOp's sensitivity factors to $10^{-9}$ relative on IEEE 57 and on the
  node-breaker example grid, and which choices did the papers leave unspecified (slack handling in the BSDF, sign of
  the phase-shift column, the three singularity thresholds, the diagonal of the LODF)?
- **Baseline.** ToOp `ba1874e`: `dc_solver/jax/{bsdf,lodf,multi_outages}.py` through `run_solver_symmetric` with a
  pass-through aggregation (Chapter 19), and `numpy_reference.py` as a second oracle.
- **Metrics.** Largest absolute and relative difference per matrix (PTDF, LODF columns, MODF for every pair of
  outages ToOp enumerates, BSDF-updated PTDF for every action of the action set); the count of "underspecified
  choices" found, each documented by a failing property test before the choice is made; coverage of the clean-room
  code.
- **Budget.** Two grids (IEEE 57 from `data/case57_data_powsybl`, the node-breaker example after a fresh
  preprocessing), every single action and every single and double outage of each, which is exhaustive, so no
  sampling; the Hypothesis properties of Chapter 33 on random meshed grids with 200 examples each as the third leg.
- **Seeds and repetitions.** The comparison is deterministic; the only randomness is in the Hypothesis strategies,
  run with three fixed seeds (`derandomize` plus `seed=1, 2, 3`) and reported with the shrunk counterexample when
  one is found.
- **Statistical test.** None is needed for a deterministic comparison; the report states the maximum differences
  and the tolerance they are judged against ($10^{-9}$ relative for double precision, justified as in Chapter 33,
  Exercise 5), and a difference above the tolerance is a finding, not a failure.
- **The figure.** One panel per matrix: the distribution (log scale) of relative differences between the clean-room
  code and ToOp over all actions and outages of both grids, with the tolerance as a vertical line and the
  underspecified cases marked, plus a table of the choices the papers leave open and what ToOp does for each. The
  deliverables that follow are the property tests ToOp could adopt and the issues for the papers' authors.
