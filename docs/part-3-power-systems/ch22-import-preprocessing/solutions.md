# Chapter 22 – Solutions

Solutions of the Lab and the Exercises of [Chapter 22](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`grid_pipeline_toop.py`](src/grid_pipeline_toop.py) (the Lab, runs with
ToOp's interpreter), [`action_count.py`](src/action_count.py) and [`bridge_lookup.py`](src/bridge_lookup.py)
(Exercise 1), [`node_reduction.py`](src/node_reduction.py) (Exercise 2), [`loadflow_limits.py`](src/loadflow_limits.py)
(Exercise 3), [`se_case14.py`](src/se_case14.py) (Exercise 4) and [`dc_state_estimation.py`](src/dc_state_estimation.py)
(Exercise 5). Each module's `main()` prints every number quoted below; the ToOp numbers were produced with ToOp
`ba1874e` on CPU (Windows, outside the container).

## Lab

`python grid_pipeline_toop.py <folder>` runs the six steps; the functions named below are in that module and
`lab_toop.ipynb` calls them. Running ToOp in place would rewrite `data/grid_node_breaker/grid/`, so everything
happens in a copy.

1. **Import and preprocess.** `import_grid()` copies `grid.xiidm` (five 225 kV node-breaker voltage levels, nine
   lines of equal reactance, three generators, two loads, 106 switches), builds `prepare_importer_parameters(grid_file,
   data_folder)` with `area_settings.cutoff_voltage = 10` and runs `convert_file`. The importer writes `grid.xiidm`,
   `importer_auxiliary_data.json`, `initial_topology/asset_topology_master_data.json` and
   `initial_topology/original_gridfile/grid.xiidm`, `loadflow_parameters.json`, the `masks/` folder and
   `nminus1_definition.json`, in about two seconds. `run_preprocessing(importer_parameters, data_folder,
   PreprocessParameters(), is_pandapower_net=False)` (or `funnel()`, which calls `load_grid` directly) adds
   `action_set.json`, `action_set_diffs.hdf5`, `static_information.hdf5` and `static_information_stats.json`.
   During import the log says why six lines leave the reward mask (`excluded_branch_ids=['L4', ..., 'L9']
   reason=missing_operational_limits`: only L1–L3 carry a permanent limit, 90 A) and warns that VLevel5 has no
   couplers.
2. **Inspect the folder.** `mask_counts()` tabulates the `True` entries: `relevant_subs` 2 of 5 buses (VL2_0 and
   VL3_0, the two voltage levels with at least two busbars and four branches; `importer_auxiliary_data.json`
   names them under `id_lists.relevant_subs`), `line_for_nminus1` 9, `line_for_reward` 3, `line_disconnectable`
   9, `generator_for_nminus1` 3, `load_for_nminus1` 2, `busbar_for_nminus1` 8 of 11 busbar sections,
   `switch_for_nminus1` 106 (every switch), `switch_for_reward` 0; the 17 other masks are all `False` or empty
   (no transformers, tie lines or boundary lines). `nminus1_definition.json` has 5 monitored elements (the three
   lines with limits and the two relevant buses) and 129 contingencies: `BASECASE`, 9 lines, 3 generators, 2
   loads, 8 busbars and 106 switches. `static_information_stats.json` after preprocessing: `n_relevant_subs` 1,
   `n_actions` 10, `n_monitored_branches` 3, `n_nminus1_cases` 11 (8 branch outages, 3 injection outages; the
   switch and busbar contingencies are not part of the DC solver's N-1), `overload_energy_n0` 0.0 and
   `overload_energy_n1` 61.44 MW. Counting actions per `bus_group_id` needs both files: `action_set.json`
   stores the stations without their actions, which live as diffs in `action_set_diffs.hdf5`;
   `load_action_set(json, hdf5)` expands them (`actions_per_station()`). The only relevant
   station, VL2_a, has $d = 5$ branches (L1, L2, L6, L7, L8) and 10 actions against the bound
   $2^{d-1} - d = 11$; the eleventh assignment is removed by the BSDF/LODF filter (step 4). VL3_a has 4 branches
   (bound 4), VL4_a 3 and VL5_a 1, and none of them is relevant after preprocessing.
3. **Watch the funnel.** `funnel()` passes `record(stage, message, *, stats=None)` to `load_grid` and turns the
   records into a table (28 stages carry `stats`, 9 more do not). Nothing changes in `n_branches` (9). The
   stages that remove the most: `assert_network_data` for the relevant substations (2 → 1: the log says
   `Removed 1 from relevant nodes. Reason: Less than 4 non-bridge branches connected ... VL3_0`; the filter runs
   in `filter_relevant_nodes` and the drop shows up in the stats of the next stage), `reduce_branch_dimension`
   for the disconnectable branches (9 → 8: L9, the bridge to VL5, left the outage masks one stage earlier and now
   leaves the disconnectable set), and `enumerate_station_realizations` for the actions (0 → 10, the only stage
   that creates them). `n_nodes` grows, 5 → 6 → 7: `combine_phaseshift_and_injection` adds the reduced-node
   column and `add_bus_b_columns_to_ptdf` the bus-B column of the relevant station. The whole `load_grid` takes
   4 to 5 s here, most of it JAX compilation.
4. **Change preprocessing.** `variants()`:

    | Parameters | `n_actions` | Seconds |
    |---|---|---|
    | default | 10 | 4.7 |
    | `action_set_filter_bsdf_lodf=False` | 11 | 4.0 |
    | `action_set_filter_bridge_lookup=False` | 10 | 4.3 |
    | `action_set_clip=2**3` | 3 or 4 | 4.4 |
    | `electrical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=1)` | 5 | 4.1 |

    The BSDF/LODF filter removes exactly one of the eleven assignments of VL2 (a split after which some N-1 case
    islands the grid); the bridge lookup removes nothing, since no branch at VL2 is a bridge. Clipping to $2^3$
    draws 8 of the 16 patterns with replacement, unseeded, so the count differs between runs (3 in one run, 4 in
    another), the point made in the worked example. A reassignment limit of one branch keeps the unsplit
    configuration and the four single-branch moves that survive the other filters. The times are dominated by
    JAX compilation, not by the filters, on a grid this small. Note that every `load_grid` run rewrites the
    artifacts of the folder: after the last variant, `action_set.json` and `action_set_diffs.hdf5` hold five
    actions.
5. **Change import rules.** `import_rules()`:

    | `RelevantStationRules` | Relevant after import | After preprocessing | Actions |
    |---|---|---|---|
    | default (2 busbars, 4 branches, 4 elements) | VL2_0, VL3_0 | 1 | 10 |
    | `min_busbars=3` | VL2_0 | 1 | 10 |
    | `min_connected_branches=2` | VL2_0, VL3_0 | 1 | 10 |

    VL3 has two busbars and four branches (L3, L6, L7, L9) but L9 is a bridge to the radial VLevel5, so it has
    three non-bridge branches and `filter_relevant_nodes` drops it whatever the import rule said. Lowering the
    import threshold below four cannot add relevant substations because preprocessing applies its own, stricter
    rule (four non-bridge branches, an asset topology, no double connections, at least one action left); the
    import rule can only remove candidates earlier, as `min_busbars=3` does with VL3. VL4 (two busbars, three
    branches) and VL5 (one busbar) never qualify.
6. **A larger grid.** `create_complex_grid_battery_hvdc_svc_3w_trafo()` (40 voltage levels at 380, 220, 110, 70
   and 63 kV; 44 lines, 10 two-winding and 4 three-winding transformers, 46 buses) imported with cutoff 10 kV has
   12 relevant substations after import and the funnel 47 nodes, 67 branches, 51 disconnectable branches, 3
   controllable PSTs at `compute_bridging_branches`; `filter_relevant_nodes` leaves 9 substations,
   `reduce_branch_dimension` 40 and `compute_branch_topology_info` 24 disconnectable branches,
   `enumerate_station_realizations` creates 301 actions and `remove_relevant_subs_without_actions` keeps 127 (the
   station VL_NL_380_a with 9 branches has 14 of its 247 possible assignments left; VL_2W_MV_HV_MV_a 11 of 120;
   VL_3W_HV_a 45 of 57). The final stats: 60 nodes, 67 branches, 9 relevant substations, 127 actions, 24
   disconnectable branches, 3 PSTs, 66 N-1 cases, 50 monitored branches, `overload_energy_n0` 36.7 MW and
   `overload_energy_n1` 1 310.9 MW; `load_grid` takes 11 to 22 s. With `cutoff_voltage = 220` the importer keeps
   6 relevant substations (the 380 kV ones) and preprocessing 5 with 100 actions: ToOp's usual 220 kV cutoff
   does not empty the search space of this grid, whose switchable stations are at 380 kV. `cutoff_voltage =
   400` does: 0 relevant substations after import, `n_relevant_subs == 0` and `n_actions == 0` after
   preprocessing, and the run completes without an error, which is exactly why the checklist exists.

## Exercise 1

Enumeration: $2^{d-1} = 32$ bit patterns for $d = 6$ (each A/B assignment once, up to swapping the busbars).
Isolation removal drops the patterns with one branch alone on a busbar, $\binom{6}{1} = 6$ of them, leaving
$2^{5} - 6 = 26$: the unsplit configuration, 15 splits of type 2|4 and 10 of type 3|3. The bridge lookup requires
at least two *non-bridge* branches on every busbar in use. With one bridge $b_0$, the splits that fail are those
whose busbar holding $b_0$ contains exactly one other branch: $b_0$ paired with each of the five others, 5
assignments (`bridge_lookup.funnel()` lists them as b0+b1, b0+b2, b0+b3, b0+b4, b0+b5). The 3|3 splits with the
bridge keep two non-bridge branches beside it and pass. So 32 → 26 → 21 actions, the unsplit one included (20
splits). The reason: after the outage of a non-bridge branch on that busbar, the bridge would be the busbar's only
link, so the LODF of that outage is undefined (the busbar islands); the lookup removes these cases without
running a single load flow, and `is_valid_bsdf_lodf` would otherwise catch them at a much higher price.

## Exercise 2

`node_reduction.py` uses a ten-bus ring with two chords (unit reactances), reference injections summing to zero
and four significant buses {0, 2, 5, 7}. `reduce_ptdf()` keeps the four PTDF columns and appends one column,
$\text{PTDF}_{\cdot,\text{insig}}\,p_\text{insig}$, the flow every branch receives from the six collapsed buses at
their reference injections; the reduced injection vector is $(p_0, p_2, p_5, p_7, 1)$. At the reference the
reduced model reproduces all twelve flows to $10^{-14}$ MW: the full product $Hp$ was simply split into two sums
and the second one precomputed. After moving 20 MW of load to collapsed bus 3 (balanced by the slack) the reduced
model is wrong by up to 11.5 MW on a branch, because the precomputed column still carries bus 3's old injection;
a change at significant bus 2 is reproduced exactly. This is acceptable for ToOp because the optimizer never
changes injections: per timestep they are inputs, fixed when `load_grid` builds the static information, and only
the topology at relevant substations varies, which touches the kept columns (and their bus-B copies). The
reduction trades a column per insignificant bus for one column, which is what makes the PTDF of a continental grid
fit on a GPU; the price is that a new timestep needs a new precomputed column, which is why the column is stored
per timestep.

## Exercise 3

For $I_\text{old} = 1\,000$ A, $f = 1.2$, $m = 0.05$ the rule
$I_\text{new} = \operatorname{clip}(f I_\text{lf},\ \min(I_\text{old}, I_\text{lf} + m I_\text{old}),\ I_\text{old})$
has three regimes (`loadflow_limits.breakpoints()`): below $I_\text{lf} = m I_\text{old}/(f-1) = 250$ A the
lower clip is active and $I_\text{new} = I_\text{lf} + 50$ A; between 250 A and $I_\text{old}/f = 833.3$ A the
factor rules, $I_\text{new} = 1.2\,I_\text{lf}$; above 833.3 A the upper clip returns the old limit, 1 000 A.
With the N-1 factor $f = 1.4$ the breakpoints move to 125 A and 714.3 A. The plot is a broken line through
(0, 50), (250, 300), (833.3, 1000) and flat afterwards, always below the old limit. A border line that is unloaded
in the base case gets $I_\text{new} = m I_\text{old} = 50$ A, 5 % of its rating: ToOp will treat any flow above 50 A
on it as an overload, which is the intended behaviour (the neighbouring TSO's lines are not to be loaded by this
TSO's remedial actions) but also a trap when a line happens to be out of service or lightly loaded in the
forecast. `new_limit(100) = 150`, `new_limit(500) = 600`, `new_limit(900) = 1000`.

## Exercise 4

`se_case14.py` builds 82 measurements on `case14` after `runpp`: 14 voltage magnitudes ($\sigma = 1\,\%$), p and q
at every bus and at the from (hv) side of the 15 lines and 5 transformers ($\sigma = 2\,\%$ of the value, at least
0.1). Two pandapower conventions matter: bus powers are consumption (`res_bus`, load positive) and must exclude
the shunt at bus 8, which the estimator models as a network element (with the shunt included, even noise-free data
give 0.005 pu of voltage error and a failed chi-square test). With exact data the estimate reproduces the power
flow to $10^{-11}$ pu and chi2 is `False`; with noisy data (seed 14) the largest errors are 0.0015 pu and 0.09°,
and `chi2_analysis` passes. Scaling the active-power measurement of line 0 (measurement 42, from side) by 1.3
makes `chi2_analysis` return `True`; the largest normalized residual is that measurement's, $r_N = 15.2$ (the
next one is −3.1), and `remove_bad_data(rn_max_threshold=3)` drops exactly measurement 42, after which the test
passes again. The topology error opens line 8 (buses 5–11) in the model while the measurements still describe
it in service: chi2 is `True` and the six largest normalized residuals all touch the opened line: its own p and q
measurements (48.9 and 25.5: the estimator predicts zero flow on a line it believes open, the meters say 7.6 MW),
the injections at bus 11 (37.2 and 13.0) and the flows of line 13 (12–13; 30.1 and 11.9), the path that now has
to feed bus 11 in the model. A single bad meter produces one dominant residual; a wrong breaker status produces a
cluster of residuals around one station, which is how topology errors are told from bad data. The pandapower
estimation code emits pandas' `SettingWithCopyWarning` and, under Python 3.14, a `DeprecationWarning` for
`~bool`; `quiet_estimation()` silences both around the library calls, since the test suite treats warnings as
errors.

## Exercise 5

For the three-bus ring of Chapter 16 ($x = 0.1, 0.1, 0.2$, $p = (100, 0, -100)$ MW, slack bus 0) the measurement
model is $z = H\theta + e$ with $\theta = (\theta_1, \theta_2)$, five rows (the flows $b_k(\theta_f - \theta_t)$ of
the three branches and the injections at buses 1 and 2, i.e. rows of $B_\text{bus}$) and $\sigma = 2$ MW for the
flow meters, 3 MW for the injection meters. `wls()` solves the normal equations
$\hat\theta = (H^\top W H)^{-1} H^\top W z$, $W = \operatorname{diag}(1/\sigma_i^2)$, and returns the residuals
$r = z - H\hat\theta$ and $J = \sum_i r_i^2/\sigma_i^2$. Noise-free, the true flows (50, 50, 50) and injections
(0, −100) are recovered exactly with $J = 0$. With one noisy sample (seed 16): $z = (48.8, 51.3, 52.1, 3.1,
-94.5)$, $\hat\theta = (-4.85, -9.86)$ against $(-5, -10)$, residuals $(0.33, 1.17, 2.79, 1.48, 4.83)$ and
$J = 5.16$, below the 95 % chi-square threshold with $m - n = 3$ degrees of freedom, $\chi^2_{0.95,3} = 7.815$:
the model and the data agree. Adding a 30 MW error to the first flow meter gives $J = 110.0$, far above the
threshold, and the largest normalized residual points at measurement 0. Over 2 000 noisy samples the mean of $J$
is 3.0, the expected value of $\chi^2_3$ (`test_dc_state_estimation.py`), which is the justification of the
test: $J$ is a chi-square sample only when $H$ is the right model and the $\sigma_i$ are right.

## Exercise 6

| Checklist item | Read first | One cause from this chapter |
|---|---|---|
| `n_relevant_subs == 0` | `static_information_stats.json` → `n_relevant_subs`; then `masks/relevant_subs.npy` and `id_lists.relevant_subs` in `importer_auxiliary_data.json` to see whether the import or the preprocessing removed them | `cutoff_voltage` above the switchable stations (Lab step 6: 400 kV on a 380 kV grid), or stations whose fourth branch is a bridge (Lab step 5: VL3), or a UCTE model without busbar/switch data so `get_switchable_buses_ucte` finds nothing |
| `n_actions == 0` | `n_actions` in the stats; `action_set.json` plus `action_set_diffs.hdf5` for the per-station count; the funnel at `enumerate_station_realizations` and `remove_relevant_subs_without_actions` | every enumerated split fails the bridge lookup or the BSDF/LODF filter (a station whose branches are all bridges or whose splits island the grid under some N-1), `action_set_clip` or `ReassignmentLimits` too small, or no coupler in the asset topology (the VLevel5 warning) so no physical realization exists |
| `n_monitored_branches == 0` | `n_monitored_branches`; `masks/line_for_reward.npy`, `trafo_for_reward.npy`; `nminus1_definition.json` → `monitored_elements` | branches without operational limits are dropped from the reward mask (`get_element_has_limits_mask`: six of nine lines here), a `view_area` that matches no element, or a black list that removed them |
| `overload_energy_n0 == overload_energy_n1 == 0` | the two fields in the stats; then `max_mw_flows` against the base-case flows in the funnel's `compute_base_loadflows` | a missing limit filled with 99 999 MW, white-list percentages that scale limits up, loadflow-based limits that are the old limit for every loaded line, or a model whose load flow did not converge and was accepted with `fail_on_non_convergence=False` so the flows are not the operating point |

In every case the first number is in `static_information_stats.json` and the second question is *which stage*
set it, which is what the `status_update_fn` callback of Lab step 3 answers.
