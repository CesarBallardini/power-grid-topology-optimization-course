# Chapter 35. Capstone: reading ToOp end to end

**Weeks:** 5 · **Semester:** 5 · **Prerequisites:** [Chapter 18](../part-3-power-systems/ch18-sensitivity-factors-2.md), [Chapter 22](../part-3-power-systems/ch22-import-preprocessing.md), [Chapter 28](../part-4-optimization/ch28-topology-optimization.md), [Chapter 29](../part-4-optimization/ch29-learning-based-control.md), [Chapter 33](../part-5-computing/ch33-testing-debugging-profiling.md), [Chapter 34](../part-5-computing/ch34-pipeline-engineering.md)

!!! info "Why it matters for ToOp"

    The capstone is where comprehension becomes contribution. Teams read ToOp end to end (papers, notebooks,
    code), test what its documentation claims against what its code does, and complete one project that answers a
    research question or improves the engine. Every team also writes the document an operator would actually
    read before switching: a short brief with the proposed topologies, their AC validation, the switching steps
    and before/after single-line diagrams. The skills transfer to any topology optimization software: read the
    paper, run the pipeline, distrust the docs, measure, and explain the result to the person who has to act on it.

## Topics

- Reading a research code base in layers: papers → runnable notebooks → a traced call path through the code.
- Critical reading: documentation, docstrings and parameters that disagree with the code; turning each
  discrepancy into an issue, a test or a pull request.
- A research question, a reproducible experiment ([Chapter 26](../part-4-optimization/ch26-experimental-methodology.md))
  and an honest comparison against a baseline.
- Contributing upstream: fork, branch, Conventional Commits pull request title, DCO sign-off, pre-commit hooks and
  CI (`docs/contribution_guide.md`, `.github/workflows/validate_pr_title.yaml`, `.github/workflows/ci.yaml`).
- Communicating with operators: from index lists (`actions`, `disconnections`, `pst_setpoints`) to switch
  operations (`orao_summary.json`), AC metrics (`ac_metrics.json` vs `unsplit_ac_metrics.json`) and single-line
  diagrams.

## Learning outcomes

- Trace one candidate topology from the genetic operators through the DC solver, the repertoire and the AC
  validation, naming the function and data structure at every step.
- Verify or refute a claim in ToOp's documentation against the code, and back the verdict with a test or a
  minimal reproduction.
- Design, run and report a reproducible experiment that answers a stated research question.
- Write a 1–2 page operator brief that a control-room engineer could act on.

## Resources

- Papers: "Accelerated DC loadflow solver for topology optimization", arXiv:2501.17529 (solver); "Bus Split
  Distribution Factors", TechRxiv, doi:10.36227/techrxiv.22298950.v1 (BSDF); arXiv:2412.16164 (unified derivation of
  distribution factors); arXiv:1606.07276 (MODF); "Transmission Topology Optimization using accelerated MapElites",
  arXiv:2605.10128 (optimizer).
- ToOp documentation: <https://eliagroup.github.io/ToOp/>; `docs/architecture/`; `docs/topology_optimizer/metrics.md`;
  `docs/topology_optimizer/ac/select_strategy.md` and `docs/topology_optimizer/ac/early_stopping.md`;
  `docs/contribution_guide.md`; `docs/wall_of_shame.md` (known shortcuts and unfixed bugs). Coding
  conventions: `.github/copilot-instructions.md`.
- Contributing: GitHub, "Fork a repository"
  <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo> and
  "Creating a pull request from a fork"
  <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork>;
  Conventional Commits 1.0.0 <https://www.conventionalcommits.org/en/v1.0.0/>; commitizen
  <https://commitizen-tools.github.io/commitizen/>; pre-commit <https://pre-commit.com/>; Developer Certificate of
  Origin <https://developercertificate.org/>.
- Single-line diagrams: pypowsybl, "Network visualization"
  <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/network_visualization.html>; ToOp's
  `grid_helpers/powsybl/single_line_diagram/get_single_line_diagram_custom.py`.

## Reading order

1. **Papers:** arXiv:2501.17529 (solver), the BSDF paper, arXiv:2412.16164, arXiv:1606.07276 (MODF), then
   arXiv:2605.10128 (optimizer).
2. **Notebooks:**
    - `notebooks/example1_dc_loadflow_example.ipynb` → `notebooks/example_asset_topology_analysis.ipynb`.
    - AC N-1 contingency analysis: build the `Nminus1Definition` yourself and call `get_ac_loadflow_results`, as
      in the lab of [Chapter 19](../part-3-power-systems/ch19-operational-security.md)
      (`packages/contingency_analysis_pkg/README.md`), not `notebooks/run_ac_contingency_analysis.ipynb`, which
      does not run (see the discrepancy list).
    - `notebooks/example2_small_grid_toop.ipynb`, remembering that its busbar-outage keywords are ignored →
      `notebooks/example3_e2e_pipeline.ipynb`.
    - Optional: `notebooks/openrao.ipynb` ([Chapter 21](../part-3-power-systems/ch21-coordinated-security-rao.md));
      like the AC contingency notebook, it is not executed by CI.
3. **Code:** trace one topology from `optimizer/dc/genetic_functions/` through `dc_solver/jax/compute_batch.py`
   (`compute_bsdf_lodf_static_flows`) and `dc_solver/jax/aggregate_results.py` back to the repertoire, then to
   `optimizer/ac/select_strategy.py` and `optimizer/ac/scoring_functions.py`.

## Documentation vs code: known discrepancies

Useful for critical reading exercises and for the documentation audit project (see also
[Appendix C](../appendices/c-toop-concept-map.md)). Each item was checked against ToOp commit `ba1874e`; re-check
before you rely on it, since the code moves.

**Documentation and docstrings**

- The README's "Pareto front" is a weighted-sum MAP-Elites repertoire
  ([Chapter 27](../part-4-optimization/ch27-pareto-multi-objective.md)).
- "(G)LODF" in the docs corresponds to LODF (single outage) and MODF (multi-outage) in the code.
- The MODF citation differs between the dc_solver README (arXiv:1606.07276) and `dc_solver/jax/multi_outages.py`
  (doi:10.1109/TPWRS.2009.2023273).
- `docs/dc_solver/switching_distance.md` describes a graph edge-cut algorithm; the code enumerates all
  coupler states (up to 20 couplers) and keeps those giving exactly two components
  (`dc_solver/preprocess/preprocess_switching.py`). The same page says a multi-connected asset keeps the busbar
  with the most other branches, while `fix_multi_connected_without_coupler` (`grid_helpers/asset_topology_helpers.py`)
  removes the connection to the lower-index busbar.
- The package documentation lists three AC evolution operators; `optimizer/ac/evolution_functions.py`
  implements only `pull`.
- `docs/usage.md` shows worker launch commands (`python3 worker.py --processed_gridfile_folder=...`) that the worker
  modules do not implement: `optimizer/dc/worker/worker.py` and `optimizer/ac/worker.py` have no CLI, and their
  `main()` expects ready-made Kafka producers, consumers and file systems (see their signatures, and "What is not
  included" in the course fork's `docker/README.md`).
- `docs/dc_solver/quickstart.md` calls `run_solver(topology, None, injection, static_information)`; the current
  signature takes `(topologies, disconnections, injections, dynamic_information, solver_config)`.
- `docs/topology_optimizer/metrics.md` itself states that the AC stage counts `switching_distance` differently from
  the DC stage ("This is a bug, not a feature"), so DC and AC switching distances of the same topology are not
  comparable.
- `packages/dc_solver_pkg/README.md` says to set `static_information.solver_config.cross_coupler_flow = True`, but
  `SolverConfig` (`dc_solver/jax/types.py`) has no such field (on a plain dataclass the assignment silently adds an
  unused attribute). The batch path always computes N-0 flows through `compute_cross_coupler_flows` from
  `unsplit_flow` (`dc_solver/jax/compute_batch.py`), and `LoadflowSolverParameters.cross_coupler_flow`
  (`optimizer/interfaces/messages/dc_params.py`) is not read anywhere else.
- The `PowsyblBackend` class docstring assumes "no HVDC lines" and "no batteries", yet `_get_injections` gathers
  batteries and LCC/VSC converter stations (`dc_solver/preprocess/powsybl/powsybl_backend.py`).
- The docstrings of `compute_cross_coupler_flows` and `_gather_bus_a_injection` (`dc_solver/jax/cross_coupler_flow.py`)
  say `True` means busbar A, but the code, consistently with `get_bus_data` (`dc_solver/jax/bsdf.py`) and
  `get_injection_per_bus` (`dc_solver/jax/injections.py`), treats `False` as busbar A and `True` as busbar B.
- The `perform_outage_single_busbar` docstring (`dc_solver/jax/busbar_outage.py`) says load flows are set to zero when
  both attempts fail; the code sets them to NaN.
- `docs/benchmark.md` points to `toop-engine-benchmark/config/` and `assess_benchmark.py` (the real names are
  `configs/` and `assess_benchmarks.py`) and sweeps `ga_config.split_subs`, which is not a field of
  `BatchedMEParameters`.
- `docs/topology_optimizer/metrics.md` calls the "exponential" overload metric exponential although it is a power
  law, describes N-1 overload energy per timestep although the code takes the worst contingency per branch, and
  states that `bb_outage_penalty` is zero when the topology is not worse than the baseline, which only holds with
  `clip_bb_outage_penalty` enabled (default off) ([Chapter 19](../part-3-power-systems/ch19-operational-security.md)).
- `docs/dc_solver/preprocessing.md` lists the preprocessing stages in a different order from the code, says relevant
  stations need 4 branches (the code requires 4 non-bridge branches), describes a one-hop node reduction (the code
  keeps two hops), and lists asset-topology files that a real import does not write
  ([Chapter 22](../part-3-power-systems/ch22-import-preprocessing.md)).
- The action-set clipping docstring mentions `2**20` actions; the default `clip_to_n_actions` is `2**23`
  (`dc_solver/preprocess/action_set.py`).
- Parallel PST group docstrings say groups are identified "during importing"; they are built in the dc_solver
  PowSyBl helpers (`dc_solver/preprocess/powsybl/powsybl_helpers.py`).
- The loadflow-based artificial limits are named `loadflow_based_n0`/`loadflow_based_n1` in the code, not
  `border_limit_*` as the `AreaSettings` docstrings say, and the `LimitAdjustmentParameters` docstring
  states its comparison the wrong way round (`importer/pypowsybl_import/loadflow_based_current_limits.py`).
- The `PreprocessParameters` docstrings omit the 90–100 % dead zone of double limits
  (`dc_solver/jax/aggregate_results.py`), and the `changing_switches_to_orao_dict` docstring shows PST and
  `TERMINALS_CONNECTION` entries in the OpenRAO summary; `optimizer/ac/summary.py` emits only `SWITCH` entries and
  never exports PST setpoints.
- The relay example in `docs/contingency_analysis/cascade.md` (80° characteristic) builds a self-intersecting
  polygon, which shapely reports as invalid ([Chapter 20](../part-3-power-systems/ch20-protection-short-circuit.md)).

**Notebooks and parameters that are silently ignored**

- `notebooks/run_ac_contingency_analysis.ipynb` does not run: it passes `GridElement` where
  `Nminus1Definition.monitored_elements` requires `MonitoredElement`, and its pandapower cell lacks a comma
  (`method="ac" n_processes=15`). CI never notices, because `notebooks/tests/test_notebooks.py` only globs
  `example*.ipynb`.
- `notebooks/example2_small_grid_toop.ipynb` and `toop-engine-benchmark/benchmark_toop.py` pass `enable_bb_outage`
  and `bb_outage_as_nminus1` to `PreprocessParameters`, which has no such fields and silently drops them; the
  preprocessing switch is `preprocess_bb_outages`, and the other two belong to the optimizer's `ga_config`.
- `compute_metrics_single_timestep` (`optimizer/ac/scoring_functions.py`) passes the misspelled keyword
  `worst_k_contingent_cases` to `Metrics`, whose field is `worst_k_contingency_cases`; the value is silently dropped.
- `early_stopping_non_convergence_percentage_threshold` (`optimizer/interfaces/messages/ac_params.py`) and
  `CLIArgs.checkpoint_frequency` (`optimizer/dc/main.py`) are defined but never read; `res_<epoch>.json` is written
  at every epoch.

**Behavior worth an issue**

- The PowSyBl backend negates phase-shift angles with the comment "TODO find out where this minus comes from..."
  (`get_shift_angles` in `dc_solver/preprocess/powsybl/powsybl_backend.py`); the sign convention is undocumented
  ([Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md)).
- `run_solver` (`dc_solver/jax/topology_looper.py`) builds a `DefaultAggregateOutputFn` that stores
  `max_mw_flow` and `branches_to_fail` and hashes as `hash(solver_config)`. With the same `SolverConfig` object, a
  new `DynamicInformation` with different branch limits reuses the compiled program and returns results based on
  the old limits ([Chapter 32](../part-5-computing/ch32-jax-in-toop.md) lab).
- `update_n0_for_pst_taps` (`dc_solver/jax/pst.py`) discards the incoming N-0 flows, which the batch path derives
  from `unsplit_flow`, and recomputes them as PTDF·P, so any offset carried by `unsplit_flow` is lost when PST taps
  are optimized.
- The 1×1 case of `solve_and_check_det` (`dc_solver/jax/unrolled_linalg.py`) checks `a != 0` exactly, while the
  other paths use tolerances (LODF `1e-11`, BSDF `1e-5`, determinant `1e-10`); a bridge whose denominator is
  numerically tiny but not zero can pass.
- `plot_repertoire` (`optimizer/dc/repertoire/plotting.py`) fills empty repertoire cells with 1.05 times the minimum
  fitness, so empty cells look like very bad topologies rather than missing ones.
- `choose_max_mw_flow` (`dc_solver/jax/aggregate_results.py`) places `critical_branch_count_limited_n_1` in the N-0
  group, so `critical_branch_count_limited_n_0` is evaluated against the unlimited rating.
- The DC N-1 oracle test `test_nminus1_results_one_timestep` is skipped, so only N-0 DC flows are cross-checked
  against an independent load flow ([Chapter 33](../part-5-computing/ch33-testing-debugging-profiling.md)).
- The import converts low-impedance lines to breakers only for the hard-coded UCTE area `"D8"`
  (`importer/pypowsybl_import/preprocessing.py`).

## Projects (one per team)

Every project states one research question, goes beyond the labs of
[Chapter 27](../part-4-optimization/ch27-pareto-multi-objective.md), and follows the experimental methodology of
[Chapter 26](../part-4-optimization/ch26-experimental-methodology.md) (budgets, seeds, repeated runs, statistics).

1. **Clean-room solver.** *Question: can the published papers alone reproduce ToOp's sensitivity factors, and where
   are they underspecified?* Re-implement PTDF, LODF, MODF and BSDF in NumPy from the papers only, cross-check
   against ToOp on IEEE 57 and a node-breaker grid, and add Hypothesis property tests
   ([Chapter 33](../part-5-computing/ch33-testing-debugging-profiling.md)) that ToOp could adopt.
2. **Pareto-based candidate selection for the AC stage.** *Question: does selecting AC candidates by Pareto
   dominance (with knee points or an achievement function) instead of ToOp's dominator filter raise the number of
   AC-accepted topologies per AC-second?* Implement the alternative selection next to
   `optimizer/ac/select_strategy.py`, and compare both on a realistic grid with repeated runs.
3. **Multi-objective quality-diversity inside ToOp.** *Question: does a multi-objective repertoire integrated with
   ToOp's discrete repertoire find operator-relevant trade-offs that scalar MAP-Elites misses?* Go beyond the
   stand-alone MOME lab: integrate it with the DC worker, and compare with scalar MAP-Elites by MOQD score and AC
   acceptance rate. The research survey found no published MOME application to grid topology, so a good result
   is publishable.
4. **Exact vs heuristic at scale.** *Question: up to what grid size and action space does ToOp reach within 1%
   hypervolume of the exact bi-objective front at equal wall-clock budget?* Extend the exact baseline from IEEE 14/30
   line switching to IEEE 118 and a node-breaker model with busbar splitting; compare AUGMECON2, NSGA-II and ToOp
   itself with anytime curves and ECDFs.
5. **DC–AC gap study.** *Question: which DC modeling assumptions best predict AC rejection of DC-optimal
   topologies, and can a cheap correction reduce rejections?* Measure how often DC-optimal topologies fail AC
   acceptance (the lab of [Chapter 28](../part-4-optimization/ch28-topology-optimization.md) checks one run on
   the node-breaker example grid), relate failures to the DC assumptions of
   [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md), and test one correction or
   screening step.
6. **Documentation audit with real pull requests.** *Question: which discrepancies are documentation errors and
   which are code defects, and how many can be guarded by an automated test?* Verify each item above. For each
   confirmed item, open an issue, then a pull request from a fork: a branch off `main`, hooks installed on the
   host (`pre-commit install`), commits signed off (`git commit -s`), a pull request title that passes
   commitizen in `.github/workflows/validate_pr_title.yaml` (for example `docs: fix run_solver signature in
   quickstart`), a test where the item is a code defect, and green CI (`.github/workflows/ci.yaml`, 90% coverage).
   Remember that the sensitive-file hook blocks `.json`, `.xiidm` and `.hdf5` fixtures unless whitelisted.
7. **HVDC setpoint actions (improve ToOp).** *Question: how much N-1 overload energy can HVDC setpoint changes remove
   on their own and combined with topology, and does the DC estimate hold in AC?* `HVDCRange` exists in
   `interfaces/stored_action_set.py` but is "currently not implemented yet in the solver", and
   `dc_solver/preprocess/network_data.py` always writes `hvdc_ranges=[]`. Model a link as a pair of injections
   ([Chapter 11](../part-2-circuits/ch11-transformers-psts-hvdc.md)) whose flow sensitivities are PTDF column
   differences, extract ranges from the PowSyBl converter stations, carry setpoints as traced data next to the PST
   taps (`dc_solver/jax/pst.py`, `dc_solver/jax/nodal_inj_optim.py`), add a mutation operator, and validate in AC
   with pypowsybl on a grid with HVDC links.

## Operator brief (required deliverable)

Every team, whatever its project, hands in a 1–2 page brief for one congested grid situation, written for a
control-room engineer rather than for a programmer:

- **Situation:** base-case and worst N-1 overloads before any action (from `unsplit_ac_metrics.json` and the AC
  load flow results), and the data date and grid model used.
- **Proposed topologies:** at most three, chosen by a stated criterion (for example the best AC-accepted topology per
  number of split substations), each with the overload energy removed in AC, the remaining critical branches,
  non-converging contingencies and voltage-angle differences (`ac_metrics.json`).
- **Switching steps:** the ordered switch operations per topology from `orao_summary.json`, named by station and
  switch, with a feasibility note per step (breaker vs disconnector, synchro-check and angle difference;
  [Chapter 20](../part-3-power-systems/ch20-protection-short-circuit.md)).
- **Diagrams:** before/after single-line diagrams of every split station, with the changed switches highlighted
  (`get_single_line_diagram_custom` with `highlight_grid_model_ids`).
- **Assumptions and risks:** DC–AC differences, contingencies not studied, and what would change the recommendation.

## Assessment rubric

| Criterion (weight) | Excellent | Adequate | Insufficient |
|---|---|---|---|
| **Correctness** (25%) | Results agree with an independent oracle; every discrepancy found is explained | Results plausible and partly cross-checked | Unchecked numbers or unexplained contradictions |
| **Validation** (20%) | DC findings confirmed in AC N-1; limits of validity stated | AC validation of the main result only | DC-only claims |
| **Reproducibility** (20%, [Chapter 26](../part-4-optimization/ch26-experimental-methodology.md)) | Pinned environment, seeds, configs and scripts regenerate every figure; repeated runs with statistics | Most results rerunnable; few repetitions | Single runs; results cannot be regenerated |
| **Code quality and tests** (15%, [Chapter 33](../part-5-computing/ch33-testing-debugging-profiling.md)) | Typed, documented code; reference, oracle or property tests; pre-commit and CI green | Tests for the main path only | No tests, or tests that cannot fail |
| **Communication** (20%) | Report and operator brief are concise, correct and actionable; diagrams readable | Clear report; brief lacks steps or diagrams | Hard to follow; no usable brief |

## Lab

!!! example "Lab: a traced, reproducible baseline and a first operator brief"

    **Tools:** the ToOp container; Jupyter + pandas + matplotlib; pypowsybl single-line diagrams; git.

    1. **Baseline run.** Run the staged pipeline of `notebooks/example3_e2e_pipeline.ipynb` (`run_preprocessing`
       once, then `run_dc_optimization_stage` and `perform_ac_analysis`) on the node-breaker example grid
       `data/grid_node_breaker/grid.xiidm` with a fixed seed and runtime, three times. Store the configuration,
       `res.json` and the AC outputs of each run in your team repository, and tabulate `initial_fitness`, `max_fitness` and AC acceptance per run.
    2. **Trace.** Pick the best DC topology and follow it by hand: its `actions` and `disconnections` indices in
       `action_set.json`, the BSDF and MODF updates in `dc_solver/jax/compute_batch.py`, the metric in
       `dc_solver/jax/aggregate_results.py`, its repertoire cell, the AC selection in `optimizer/ac/select_strategy.py`
       and its AC score in `optimizer/ac/scoring_functions.py`. Recompute its DC fitness from `res.json` as the
       negated weighted sum of `target_metrics`.
    3. **Independent AC check.** Rebuild the AC N-1 result of that topology, either with
       `get_ac_loadflow_results` on `topology_N/modified_network.xiidm` and an `Nminus1Definition` built with
       `MonitoredElement` (as in the [Chapter 19](../part-3-power-systems/ch19-operational-security.md) lab), or
       with `create_loadflow_runner(data_folder, grid_file_path).run_ac_loadflow(actions, disconnections)`
       (`optimizer/benchmark/benchmark_utils.py`), and compare with `ac_metrics.json`.
    4. **Draft brief.** Produce before/after single-line diagrams of the split station and a first operator brief
       following the template above; exchange briefs between teams for review.
    5. **Pick your discrepancy.** Verify one item of the discrepancy list with a minimal reproduction and file it as
       the first issue of your project.

## Exercises

1. *(Jupyter: pandas)* From a `res.json`, rebuild the table of `best_topos` with fitness and every `extra_scores`
   metric, and check that `fitness` equals the negated weighted sum of `target_metrics` for every row. Which
   topology would you give an operator who accepts at most one split substation?
2. *(ToOp container)* Reproduce the silent drop of `worst_k_contingent_cases` in
   `optimizer/ac/scoring_functions.py` with a two-line example, write the failing test, and draft the pull request
   title and body as they would pass `validate_pr_title.yaml`.
3. *(Jupyter: pypowsybl)* For one split station, generate the before and after single-line diagrams, highlight the
   switches listed in `orao_summary.json`, and write the switching steps in operator language.
4. *(Jupyter: NumPy + matplotlib)* For the topologies of your baseline run, plot DC switching distance against AC
   switching distance. Relate what you see to the note in `docs/topology_optimizer/metrics.md`.
5. For project 7, derive the DC sensitivity of every branch flow to a setpoint change $\Delta P$ on an HVDC link
   between buses $i$ and $j$, and show that it equals $\Delta P\,(\mathrm{PTDF}_{\cdot,i}-\mathrm{PTDF}_{\cdot,j})$ up to
   the sign convention. Where would this column live in ToOp's data structures?
6. Pick one project and write a one-page experimental plan: research question, baseline, metrics, budget, number of
   seeds, statistical test, and the figure that would answer the question.
