# Chapter 22. Import and preprocessing: from grid model to search space

**Weeks:** 2.5 · **Semester:** 4 · **Prerequisites:** [Chapter 13](ch13-substations-grid-models.md), [Chapter 16](ch16-dc-power-flow.md), [Chapter 18](ch18-sensitivity-factors-2.md), [Chapter 19](ch19-operational-security.md)

!!! info "Why it matters for ToOp"

    Most "the optimizer found nothing" reports are import or preprocessing problems: no relevant substation
    survived the rules, the action set is empty, nothing is monitored, or the base case has no congestion
    (each shows up as a zero in `static_information_stats.json`). The search space the GPU explores is decided
    long before the optimizer starts:

    - **Import** (`importer/pypowsybl_import/preprocessing.py`, `convert_file`): load UCTE, CGMES or XIIDM, find
      a converging AC load flow, apply white and black lists, compute the masks that encode policy
      (`importer/pypowsybl_import/powsybl_masks.py`), add artificial border limits, and write the processed grid
      folder (`interfaces/folder_structure.py`).
    - **Preprocessing** (`dc_solver/preprocess/preprocess.py`, `load_grid` in `dc_solver/preprocess/convert_to_jax.py`):
      bridges, relevant-node filters, PTDF and PSDF, node and branch reduction, N-2-safe disconnectable branches,
      action enumeration and filtering (`dc_solver/preprocess/action_set.py`), physical station realization
      (`dc_solver/preprocess/preprocess_station_realisations.py`), bus-B columns, and optional busbar outages.
    - **Artifacts** you can inspect: `masks/*.npy`, `nminus1_definition.json`, `action_set.json`,
      `static_information_stats.json` (`DynamicInformationStats` in
      `interfaces/messages/preprocess/preprocess_results.py`) and `static_information.hdf5`.

    Every other optimizer, not only ToOp, needs the same pipeline: a grid model is not a search space until
    someone decides what may switch, what is watched and which states are valid.

## Topics

- **The processed grid folder contract.** One folder per timestep; file names come from `PREPROCESSING_PATHS`
  and `NETWORK_MASK_NAMES` (`interfaces/folder_structure.py`), never hard-coded. The importer writes the grid
  snapshot, masks, load flow parameters, initial asset topology, auxiliary data and the first N-1 definition;
  `load_grid` adds the solver artifacts and refreshes the N-1 definition (`docs/quickstart.md`,
  `packages/contingency_analysis_pkg/README.md`).
- **Areas and masks.** `AreaSettings`: `control_area` (where switching may happen), `view_area` (monitored
  branches), `nminus1_area` (failed elements), `cutoff_voltage` (default 220 kV; equipment below it is ignored).
  Masks: `relevant_subs`, `*_for_nminus1`, `*_for_reward`, `*_disconnectable`, `*_overload_weight`,
  `*_n0_n1_max_diff_factor`, border flags. Branches without operational limits are dropped from the reward mask
  (`get_element_has_limits_mask`); the slack is removed from the relevant substations.
- **Relevant-station rules.**
    - Import, CGMES and XIIDM path: `RelevantStationRules` with `min_busbars = 2`, `min_connected_branches = 4`,
      `min_connected_elements = 4` (`interfaces/messages/preprocess/preprocess_commands.py`,
      `importer/pypowsybl_import/cgmes/powsybl_masks_cgmes.py`).
    - Import, UCTE path: a bus with more than one busbar and switches in its voltage level
      (`get_switchable_buses_ucte` in `importer/pypowsybl_import/ucte/powsybl_masks_ucte.py`).
    - Preprocessing, always: at least 4 connected **non-bridge** branches, an asset topology for the node, no
      materially split station, no asset connected to two busbars (double connection), and at least one
      action left after filtering (`filter_relevant_nodes_*`, `remove_relevant_subs_without_actions` in
      `dc_solver/preprocess/preprocess.py`).
- **Graph preprocessing** (revisits [Chapter 3](../part-1-math/ch03-graph-theory.md)). Bridges are excluded from
  the N-1 and disconnectable masks (their LODF denominator is zero, [Chapter 17](ch17-sensitivity-factors-1.md));
  `filter_disconnectable_branches_nminus2` keeps only disconnectable branches whose removal does not island the
  grid under any N-1 outage (`find_n_minus_2_safe_branches` in `dc_solver/preprocess/helpers/find_bridges.py`).
- **Dimension reduction.** `reduce_node_dimension` keeps nodes of relevant substations, of monitored, outaged,
  PST and busbar-outage branches, multi-outage nodes, the slack, and everything within two branches of them;
  the other nodes collapse into one `REDUCED_NODE` column per timestep that carries their precomputed flow
  contribution $\text{PTDF}_{\cdot,\text{insig}}\,p_\text{insig}$ with unit injection
  (`dc_solver/preprocess/helpers/reduce_node_dimension.py`). `reduce_branch_dimension` drops branches that are
  neither monitored, outaged, nor at a relevant substation. `add_bus_b_columns_to_ptdf` then appends one column
  per relevant substation, a copy of its bus A column with zero injection (`get_extended_ptdf` in
  `dc_solver/preprocess/helpers/ptdf.py`), ready for the BSDF of [Chapter 18](ch18-sensitivity-factors-2.md).
- **Action enumeration.** For a station with $d$ branches, `make_action_repo` enumerates the $2^{d-1}$
  assignments that are distinct up to swapping A and B, keeps the representation with fewer branches on B (or
  closer to a starting configuration), and removes assignments that isolate a single branch; at most
  $2^{d-1} - d$ actions remain, the unsplit one included. Above `action_set_clip` (default $2^{23}$) a random
  subset is drawn instead (`enumerate_branch_actions_for_sub`, `make_action_repo` in
  `dc_solver/preprocess/action_set.py`).
- **Validity filters.** Bridge lookup: every split must leave at least two non-bridge branches on each busbar
  (`filter_splits_by_bridge_lookup`, cheap). BSDF/LODF: apply the split, then check that the BSDF, every LODF of
  the N-1 outages and every multi-outage MODF are well defined, i.e. no N-1 case islands the split grid
  (`is_valid_bsdf_lodf`, `filter_splits_by_bsdf`, costly, on by default). Reassignment limits cap the Hamming
  distance to the starting configuration (`ReassignmentLimits`).
- **Physical realization.** Electrical actions are mapped to physical busbars and couplers
  (`enumerate_station_realisations`; `docs/dc_solver/install_time.md`); infeasible actions are dropped. Master
  data (structural, keyed by `bus_group_id`) vs runtime asset topology (current coupler states)
  (`docs/dc_solver/preprocessing.md`, [Chapter 13](ch13-substations-grid-models.md)).
- **PSTs.** Parallel PST groups (PowSyBl only) share the unordered bus pair, nominal voltage, tap range, number of
  steps and the full step table (`_build_pst_group_labels` and `_identify_pst_buckets` in
  `dc_solver/preprocess/powsybl/powsybl_helpers.py`; mask built by `dc_solver/preprocess/parallel_pst_groups.py`).
  The DC flag `pst_linear` means the reactance is constant across taps (`get_linear_pst`); tap-dependent
  susceptances are handled in the solver (`dc_solver/jax/pst.py`).
- **Limits.** `get_p_max` converts current limits with $P = \sqrt{3}\,V I$, takes `permanent_limit` for N-0 and
  the `N-1` limit (falling back to the permanent limit) for N-1, prefers `loadflow_based_n0/n1` limits where
  present, and fills a missing limit with 99 999 MW. Loadflow-based border and DSO-transformer limits
  (`importer/pypowsybl_import/loadflow_based_current_limits.py`, `LimitAdjustmentParameters`):
  $I_\text{new} = \operatorname{clip}\big(f I_\text{lf},\ \min(I_\text{old}, I_\text{lf} + m I_\text{old}),\ I_\text{old}\big)$
  with $f = 1.2$ (N-0) or $1.4$ (N-1) and $m = 0.05$. On the UCTE path, DACF/PRDx white lists scale an
  element's operational limits by a percentage and add a separate `N-1` limit where the N-0 and N-1 percentages
  differ; black-listed elements are removed from the N-1, reward and disconnectable masks and from the relevant
  substations (`importer/pypowsybl_import/dacf_whitelists.py`, `importer/pypowsybl_import/network_analysis.py`,
  `importer/pypowsybl_import/powsybl_masks.py`).
- **Model assembly.** Network reduction around the areas with PowSyBl's `reduce_by_ids_and_depths`
  (`importer/pypowsybl_import/network_reduction.py`, off by default); UCTE–CGMES merge as a separate utility
  (`docs/importer/pypowsybl/merge_ucte_with_cgmes.md`, `importer/pypowsybl_import/merge_ucte_cgmes/`).
- **Data quality in real grid models.**
    - Missing or implausible limits (see above).
    - Islands: the PowSyBl backend keeps only the main connected and synchronous component
      (`_get_nodes` in `dc_solver/preprocess/powsybl/powsybl_backend.py`).
    - Zero or low impedance: on UCTE import, lines inside one voltage level of the hard-coded area prefix `D8`
      with $x \le 0.05\ \Omega$ become breakers, and branches across switches are removed
      (`importer/pypowsybl_import/network_analysis.py`).
    - Non-converging base case: the importer tries two parameter sets with three voltage initializations
      (previous values, DC values, uniform) before failing or continuing with `fail_on_non_convergence=False`
      (`find_converging_loadflow_params`).
    - Unsupported elements: the PowSyBl backend asserts that there are no three-winding transformers and no
      shunt compensators with active power; batteries and LCC/VSC converters enter as injections.
    - Silent no-ops: unknown keyword arguments to the Pydantic parameter models are ignored (the models in
      `interfaces/messages/preprocess/preprocess_commands.py` keep Pydantic's default `extra="ignore"`).
- **State estimation and topology errors (awareness).** Measurements $z = h(x) + e$; weighted least squares
  $\min_x J(x) = \sum_i (z_i - h_i(x))^2/\sigma_i^2$ solved by Gauss–Newton with gain matrix
  $G = H^\top W H$ (revisits [Chapter 6](../part-1-math/ch06-newton-raphson.md) and the normal equations of
  [Chapter 4](../part-1-math/ch04-numerical-linear-algebra.md)); observability; bad data detection with the
  chi-square test on $J$ and the largest normalized residual; a wrong breaker status appears as a cluster of bad
  data around one station. The grid models ToOp imports are the output of this process, errors included.
- **Search-space diagnostics.** The `status_update_fn` of `load_grid` receives `NetworkDataStats` (nodes,
  branches, relevant substations, actions, disconnectable branches, PSTs) before every stage
  (`interfaces/status_update.py`, `get_network_data_stats` in `dc_solver/preprocess/network_data.py`). The final
  `DynamicInformationStats` gives the "optimizer found nothing" checklist: `n_relevant_subs == 0`,
  `n_actions == 0`, `n_monitored_branches == 0`, and `overload_energy_n0 == overload_energy_n1 == 0`.

## Learning outcomes

- Run ToOp's import and preprocessing on a node-breaker grid, inspect every artifact of the processed grid
  folder, and explain each count in `static_information_stats.json`.
- Trace the size of the search space through the preprocessing stages and predict how area settings, relevance
  rules, clipping and validity filters change it.
- Compute by hand the number of enumerated actions of a station and the effect of random clipping.
- Diagnose an empty or degenerate search space from the artifacts, and name the data-quality problems (limits,
  islands, impedances, convergence, topology errors) that cause it.

## Resources

- ToOp documentation: `docs/dc_solver/preprocessing.md`, `docs/dc_solver/install_time.md`,
  `docs/dc_solver/busbar_outage.md`, `docs/importer/pypowsybl/index.md`,
  `docs/importer/pypowsybl/merge_ucte_with_cgmes.md`, `docs/quickstart.md` (processed grid folder). ToOp code and
  notebooks: `interfaces/folder_structure.py` (folder contract), `AreaSettings` and `PreprocessParameters` in
  `interfaces/messages/preprocess/preprocess_commands.py` (importer and preprocessing parameters),
  `notebooks/example2_small_grid_toop.ipynb` and `notebooks/example3_e2e_pipeline.ipynb` with their helpers in
  `optimizer/benchmark/benchmark_utils.py` (staged preprocessing with `run_preprocessing`).
- Documentation:
  - pypowsybl network reduction (`reduce_by_ids_and_depths`) in the network user guide
    <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/network.html>.
  - pandapower state estimation (`estimate`, `chi2_analysis`, `remove_bad_data`)
    <https://pandapower.readthedocs.io/en/latest/estimation.html>.
  - pandapower diagnostic function (disconnected elements, implausible impedances, wrong switch configuration)
    <https://pandapower.readthedocs.io/en/latest/powerflow/diagnostic.html>.
- Textbooks on state estimation:
  - Grainger & Stevenson, *Power System Analysis* (1994), chapter "State estimation of power systems".
    (12 syllabi) **BORROW** [`powersystemanaly0000grai_w3w3`](https://archive.org/details/powersystemanaly0000grai_w3w3).
    ES: *Análisis de sistemas de potencia* (McGraw-Hill, 1996).
  - Ahmad, *Power System State Estimation* (Artech House, 2013): energy management systems, power flow, weighted
    least squares estimation. **BORROW**
    [`powersystemstate0000ahma`](https://archive.org/details/powersystemstate0000ahma).
  - Abur & Gómez-Expósito, *Power System State Estimation: Theory and Implementation* (CRC Press, 2004).
    (1 syllabus) **NOT ON IA**.
  - Monticelli, *State Estimation in Electric Power Systems: A Generalized Approach* (Kluwer, 1999).
    (1 syllabus) **NOT ON IA**.
  - Wood, Wollenberg & Sheblé 3rd ed., the state estimation chapter. (11 syllabi) **NOT ON IA** (3rd ed.).

## Worked example

How big is the action space of one station? The code enumerates integers $0 \dots 2^{d-1}-1$ as bit patterns, so
each A/B assignment appears once; it flips patterns with more than $d/2$ branches on B and drops patterns with
exactly one branch on either busbar.

```python
import numpy as np

def n_actions(d):
    r = np.arange(2 ** (d - 1))
    repo = ((r[:, None] >> np.arange(d)) & 1).astype(bool)
    flip = repo.sum(axis=1) > d / 2
    repo[flip] = ~repo[flip]
    on_b = repo.sum(axis=1)
    repo = repo[(on_b != 1) & (on_b != d - 1)]
    return len(np.unique(repo, axis=0))

print([(d, n_actions(d), 2 ** (d - 1) - d) for d in range(4, 9)])
# [(4, 4, 4), (5, 11, 11), (6, 26, 26), (7, 57, 57), (8, 120, 120)]
```

For $d = 4$ the three splits are the 2|2 partitions; for $d = 5$ they are the $\binom{5}{2} = 10$ partitions 2|3.
Random clipping starts when $2^{d-1} > 2^{23}$, i.e. at $d = 25$ branches. `np.random.choice` draws with
replacement and duplicates are removed, so for $d = 25$ the expected number of distinct actions is
$N\,(1 - e^{-k/N}) \approx 6.6$ million for $N = 2^{24}$ possible assignments and $k = 2^{23}$ draws, not
$2^{23}$. The draw uses NumPy's global random state with no explicit seed, so two preprocessing runs of such a station
can give different action sets.

## Documentation vs code: known discrepancies

- `docs/dc_solver/preprocessing.md` lists the `preprocess()` stages in a different order from the code (bridges
  are computed before the relevant-node filters), omits three logged stages (`assert_network_data`,
  `compute_separation_set`, `preprocess_bb_outage`) and three relevance filters, says a relevant node needs "4
  branches" where the code requires 4 non-bridge branches, and says nodes "more than one hop away" are merged
  while `get_significant_nodes` expands two hops.
- The same page lists `initial_topology/asset_topology_runtime.json` and `initial_topology/asset_topology.json`
  as importer artifacts; a real `convert_file` run writes only `asset_topology_master_data.json` and the original
  grid file (no source file of `packages/importer_pkg/src` uses the other two `PREPROCESSING_PATHS` keys).
- `enumerate_branch_actions_for_sub` documents `clip_to_n_actions` as defaulting to $2^{20}$; the default is
  $2^{23}$.
- `dc_solver/preprocess/parallel_pst_groups.py` and `PowsyblBackend._get_parallel_pst_groups` say the parallel
  PST groups are identified during importing; they are built when the PowSyBl backend loads the transformers
  (`_build_pst_group_labels` in `dc_solver/preprocess/powsybl/powsybl_helpers.py`).
- The `PowsyblBackend` class docstring assumes no HVDC lines and no batteries, but `_get_injections` includes
  batteries and LCC/VSC converters.
- `AreaSettings` docstrings call the artificial limits `border_limit_n0`/`border_limit_n1` and say
  flows are limited to the current N-0 flows; the code names them `loadflow_based_n0`/`loadflow_based_n1` and
  applies the factor-and-clip rule above. Step 3 of the `LimitAdjustmentParameters` docstring states the
  comparison the wrong way round.

## Lab

!!! example "Lab: from a node-breaker grid file to a search space"

    Tools: ToOp container and notebooks ([Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md)), Jupyter
    with NumPy, pandas and matplotlib ([Chapter 30](../part-5-computing/ch30-scientific-python.md)).

    1. **Import and preprocess.** Copy `data/grid_node_breaker/grid.xiidm` into a fresh working folder (running
       ToOp in place rewrites tracked files). Build importer parameters with `prepare_importer_parameters` and
       set `area_settings.cutoff_voltage = 10`; run `run_preprocessing(importer_parameters, data_folder,
       PreprocessParameters(), is_pandapower_net=False)` as `run_pipeline` does in
       `notebooks/example2_small_grid_toop.ipynb` (these helpers live in `optimizer/benchmark/benchmark_utils.py`).
    2. **Inspect the folder.** Load every `masks/*.npy` with NumPy and tabulate the number of `True` entries;
       read `nminus1_definition.json`, `importer_auxiliary_data.json`, `static_information_stats.json` and
       `action_set.json` with pandas. Count actions per `bus_group_id` and compare each station's count with the
       upper bound $2^{d-1} - d$.
    3. **Watch the funnel.** Re-run `load_grid(DirFileSystem(str(data_folder)), parameters=...,
       status_update_fn=record)` with a callback `record(stage, message, *, stats=None)` that appends the stage
       name and `stats` to a list. Plot `n_nodes`, `n_branches`, `n_relevant_subs`, `n_actions` and
       `n_disc_branches` per stage with matplotlib and name the stage that removes most of each.
    4. **Change preprocessing.** Repeat step 3 with `action_set_filter_bsdf_lodf=False`, with
       `action_set_filter_bridge_lookup=False`, with `action_set_clip=2**3` and with
       `electrical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=1)`. Record `n_actions`
       and wall-clock time in a table.
    5. **Change import rules.** Re-run step 1 with `relevant_station_rules=RelevantStationRules(min_busbars=3)`
       and with `min_connected_branches=2`. Explain why lowering the import threshold below four does not add
       relevant substations after preprocessing.
    6. **A larger grid.** Repeat steps 1–3 on `create_complex_grid_battery_hvdc_svc_3w_trafo()` (generated and
       imported as in `notebooks/example3_e2e_pipeline.ipynb`) and compare its `stats` with those of the
       node-breaker grid. Then set `cutoff_voltage` back to 220 kV and reproduce the first item of the
       "optimizer found nothing" checklist.

## Exercises

1. Pen and paper: a station has six branches, one of which is a bridge. List how many actions survive
   enumeration, isolation removal and the bridge-lookup filter, and which assignments the bridge lookup removes.
2. *(Jupyter: NumPy)* Implement the node reduction of `reduce_ptdf_and_nodal_injections` for a 10-bus DC grid:
   keep 4 significant buses, collapse the rest into one column, and verify that the reduced PTDF reproduces all
   branch flows exactly for the reference injections but not after changing an injection at a collapsed bus.
   Why is that acceptable for ToOp?
3. *(Jupyter: NumPy)* With the loadflow-based limit rule and $f = 1.2$, $m = 0.05$, plot the new limit against the
   loadflow current for an old limit of 1 000 A. At which current does each branch of the clip become active, and
   what happens to a border line that is unloaded in the base case?
4. *(Jupyter: pandapower)* On `pandapower.networks.case14()`, run a power flow, create voltage and power
   measurements from the results with Gaussian noise ($\sigma$ = 1 % and 2 %), run `pandapower.estimation.estimate`,
   then corrupt one flow measurement by 30 % and find it with `chi2_analysis` and `remove_bad_data`. Finally
   simulate a topology error by opening a line in the model but not in the measurements, and describe the
   residual pattern.
5. *(Jupyter: NumPy)* DC state estimation: with $z = H\theta + e$ for line flows and injections of the
   [Chapter 16](ch16-dc-power-flow.md) model, compute the WLS estimate, the residuals and $J$, and check the
   chi-square test at 95 % for $m - n$ degrees of freedom.
6. Pen and paper: for each item of the "optimizer found nothing" checklist in the Topics, name the artifact
   and field you would read first and one data-quality cause from this chapter that produces it.
