# Appendix C — ToOp Concept Map

An inventory of the domain knowledge the ToOp code relies on, with evidence of where each concept appears.
It is the basis for the chapter selection in the [study plan](../index.md) and is meant as an instructor
reference. It was produced by reading the documentation, all six packages and the notebooks of the ToOp
repository (local checkout `../ToOp`, commit `ba1874e`, September 2026).

Levels: **B** = basic (sophomore), **I** = intermediate (junior/senior EE or CS), **A** = advanced (graduate or
specialist).

## 0. Headline findings (read first)

1. **The DC solver works entirely with sensitivity factors.**
   - It builds the PTDF once, on CPU, with a scipy sparse solve.
   - Every later change is a low-rank update applied on GPU. There is no refactorization.
     - Bus splits use BSDF, a rank-1 update.
     - Branch disconnections use LODF, a rank-1 update.
     - Multi-outages (3-winding transformers, busbars) use MODF, a k×k solve.
     - PST tap and susceptance changes use a Woodbury-style correction.
   - Linear algebra is the core prerequisite.
2. **The "Pareto front" wording in README.md oversells the method.**
   - The DC fitness is a weighted-sum scalarization: `fitness = sum(-metric*weight)` in `dc/genetic_functions/scoring_functions.py:266`.
   - Diversity comes from discrete MAP-Elites descriptors (grid cells).
   - Dominance appears only as a heuristic "dominator filter" in AC candidate selection.
   - No NSGA-style non-dominated sorting, crowding distance or hypervolume exists anywhere.
3. **The docs promise more AC evolution operators than the code has.**
   - Docs (`packages/topology_optimizer_pkg/README.md`, `.github/copilot-instructions.md`) list pull, reconnect and close_coupler.
   - `ac/evolution_functions.py` implements only `pull` plus selection.
4. **"(G)LODF" is not literal.** The code has LODF (single outage) and MODF (multi-outage, the generalized form). Nothing is named GLODF.
5. **The MODF citation differs between docs and code.**
   - The dc_solver README cites arXiv:1606.07276.
   - The code in `jax/multi_outages.py` cites DOI 10.1109/TPWRS.2009.2023273 ("equation 7").
6. **Switching distance is implemented differently from how the doc describes it.**
   - `docs/dc_solver/switching_distance.md` says "graph algorithm involving edge cuts".
   - The code enumerates all 2^n coupler open/closed states (n ≤ 20) and keeps those giving exactly 2 connected components, then takes the minimum Hamming distance (`preprocess/preprocess_switching.py:make_separation_set`).
7. **Some items you asked about are absent or only nominal.**
   - Gradient optimization: none (no `jax.grad`).
   - Sharding / `shard_map`: none; only `jax.pmap`.
   - lineax: commented out.
   - evosax, optax: not used.
   - Redispatch: not modeled, only named as the motivation.
   - HVDC optimization: `HVDCRange` exists but "not implemented yet in the solver"; converter stations are plain injections.
   - OpenRAO: only an export format (`ac/summary.py`) plus a stub notebook.
8. **"Distance protection zones" belong to the contingency analysis package.** They are part of an optional pandapower cascading-failure simulator: relay impedance R/X is tested against quadrilateral polygons in the R-X plane (details in §3.9).

---

## 1. Literature references found anywhere in the repo

| # | Reference as it appears in the repo | Where |
|---|---|---|
| L1 | **Accelerated DC loadflow solver for topology optimization**, arXiv:2501.17529. Solver paper, called "our paper on load flow solving". | `packages/dc_solver_pkg/src/toop_engine_dc_solver/jax/__init__.py` (title plus URL); `packages/dc_solver_pkg/README.md`; `docs/index.md`; `.github/copilot-instructions.md` |
| L2 | DOI **10.1109/PowerTech59965.2025.11180422**, "our paper on load flow solving". The IEEE Kiel PowerTech 2025 version of L1 (Westerbeck, van Dijk, Viebahn, Merz & Witthaut); no title is given in the repo. | `README.md:145` |
| L3 | arXiv:**2605.10128**, "our paper on the optimizer architecture" / "high level academic introduction". No title in the repo; it is Westerbeck, Hilfrich & Witthaut, "Transmission Topology Optimization using accelerated MapElites" (2026). | `README.md:34,145` |
| L4 | **Bus Split Distribution Factors**, DOI 10.36227/techrxiv.22298950.v1 (TechRxiv). Also linked as https://www.techrxiv.org/users/689474/articles/681212-bus-split-distribution-factors (TechRxiv blocks many automated clients; archived copy: https://web.archive.org/web/20240614093230/https://www.techrxiv.org/users/689474/articles/681212-bus-split-distribution-factors). | `jax/bsdf.py`, `jax/__init__.py`, `tests/numpy_reference.py`, `tests/preprocessing/helper/test_ptdf.py:57` (test values taken from the paper), dc_solver README |
| L5 | **Unified algebraic deviation [sic, derivation] of distribution factors in linear power flow**, https://doi.org/10.48550/arXiv.2412.16164 | `jax/bsdf.py`, `jax/__init__.py`, `tests/numpy_reference.py` |
| L6 | DOI **10.1109/TPWRS.2009.2023273**, cited for MODF, "equation 7", notation PTDF^0_{M,O}. No title in the repo. Verified via Crossref: Guo, Fu, Li & Shahidehpour, "Direct Calculation of Line Outage Distribution Factors", IEEE Trans. Power Systems 24(3):1633–1634, 2009. | `packages/dc_solver_pkg/src/toop_engine_dc_solver/jax/multi_outages.py:89,118,247,295` |
| L7 | **MODF formulation**, arXiv:1606.07276. No title in the repo. Verified: Ronellenfitsch, Manik, Hörsch, Brown & Witthaut, "Dual theory of transmission line outages", IEEE Trans. Power Systems, 2017, doi:10.1109/TPWRS.2017.2658022. | `packages/dc_solver_pkg/README.md` (Scientific Background) |
| L8 | **PSDF**, DOI 10.1109/TPAS.1985.319195 (IEEE Trans. PAS-104:1656–1662, 1985). No title in the repo; verified via Crossref: Srinivasan, Rao, Indulkar & Venkata, "On-Line Computation of Phase Shifter Distribution Factors and Lineload Alleviation". | `packages/dc_solver_pkg/README.md` |
| L9 | JAX software citation: Bradbury, Frostig, Hawkins, Johnson, Leary, Maclaurin, Necula, Paszke, VanderPlas, Wanderman-Milne, Zhang, "JAX: composable transformations of Python+NumPy programs", 2018, http://github.com/jax-ml/jax | `README.md`, `docs/index.md` |
| L10 | UCTE data exchange format spec (ENTSO-E): https://eepublicdownloads.entsoe.eu/clean-documents/pre2015/publications/ce/otherreports/UCTE-format.pdf | `importer_pkg/.../ucte_toolset/ucte_io.py:37`, `exporter/asset_topology_to_ucte.py:44` |
| L11 | ENTSO-E, "Quality of datasets and calculations", 3rd edition (UCTE area codes): .../Publications/SOC/Continental_Europe/150420_quality_of_datasets_and_calculations_3rd_edition.pdf | `interfaces_pkg/.../messages/preprocess/preprocess_commands.py:20` |
| L12 | pandapower source used for the PTDF and B matrices: `makePTDF.py` (commit 7086cf7) and `makeBdc.py` (v2.0.1); pandapower tutorial `create_advanced.ipynb`; pandapower issues #2517 | `preprocess/helpers/ptdf.py`; `grid_helpers` example grids |
| L13 | PowSyBl docs: transformer model (cited as `www.powsybl.org/pages/documentation/grid/model/#transformers`, now offline; archived copy: https://web.archive.org/web/20240816020722/https://www.powsybl.org/pages/documentation/grid/model/#transformers); grid exchange formats; network/subnetwork; OpenLoadFlow parameters and security parameters; pypowsybl issue #1153 | `powsybl_backend.py:274`, `docs/dc_solver/loadflow_parameters.md`, contingency_analysis |
| L14 | QDax (https://github.com/adaptive-intelligent-robotics/QDax); the repertoire and MAP-Elites code is "adapted from QDax" | `topology_optimizer_pkg/.../dc/repertoire/discrete_map_elites.py`, `discrete_me_repertoire.py` |
| L15 | grid2op "topo-vect" format (the concept only, no URL) | `docs/dc_solver/quickstart.md` |
| L16 | Implementation notes: JAX issues #4258, #4572, #9940; gist tbenthompson/faae311ec4e465b0ff47b4aabe0d56b2 (unrolled 3×3 solve); JAX "Common Gotchas" (float64); StackOverflow 8317403; polars issue 27509 | `jax/unrolled_linalg.py`, `jax/utils.py`, `docs/dc_solver/quickstart.md` |
| L17 | Talks: LF Energy Summit 2025 (https://www.youtube.com/watch?v=XteDpNsX75A) and 2024 (lfenergy.org recap, "A GPU-native approach on tackling grid topology optimization") | `README.md`, `docs/index.md` |
| L18 | Tooling docs: c4model.com, likec4.dev, hydra.cc, uv docs, Material for MkDocs | `docs/architecture/README.md`, `docs/benchmark.md` |

There is no bibliography file, and `mkdocs.yml` has no references section. The nav contains only docs pages plus mkdocstrings API references.

Recommended textbooks are **not** cited anywhere in the repo (no Wood & Wollenberg, Grainger & Stevenson, Kundur, Mouret & Clune, etc.). You will need to pick them.

---

## 2. Documentation inventory (what theory the docs explain)

- `docs/dc_solver/intro.md` (includes `packages/dc_solver_pkg/README.md`): PTDF/(G)LODF/BSDF batching; static vs dynamic information (JAX recompilation); symmetric mode vs injection bruteforcing; cross-coupler flow; cost of BSDF and LODF ∝ `max_num_splits` / `max_num_disconnections`; scientific background (L1, L4, L7, L8).
- `docs/dc_solver/preprocessing.md`: the full preprocessing pipeline, 25 steps.
  - Relevant nodes need ≥4 branches (N-1 safe split).
  - PTDF, PSDF, node reduction (nodes more than one hop away are merged), bridges, N-2-safe disconnectables.
  - trafo3w multi-outages, injection outages, bus-B PTDF columns.
  - Master vs runtime asset topology; parallel PST grouping rules (same voltage, same bus pair, same tap tables).
- `docs/dc_solver/busbar_outage.md`: busbar outage as a multi-outage (MODF) plus nodal ΔP.
  - Outage propagation over closed non-breaker couplers and double-connected assets.
  - Stub or bridge branches; articulation-node busbars are excluded.
  - Physical vs electrical busbars; 2^(n-1)-1 configurations; retry with a "skeleton branch".
  - Two modes: busbar outage as N-1, or as an N-2-like penalty.
- `docs/dc_solver/switching_distance.md`: physical vs electrical switching; Hamming distance over translation sets; asset topology.
- `docs/dc_solver/install_time.md`: realizing an electrical action as a physical configuration (feasible coupler configurations, busbar_A mapping, minimum switching).
- `docs/dc_solver/loadflow_parameters.md`: distributed slack (smaller AC/DC mismatch), voltage initialization modes, convergence fallbacks.
- `docs/dc_solver/quickstart.md`:
  - Topo-vect (grid2op): each substation runs in at most a 2-node configuration.
  - Metrics table; branch weights formula (overload_grid = Σ weight·overload).
  - **Double limits** (lower 0.9 and upper 1.0 bands) and **N0-N1 max-diff factors** (limit power pushed into the distribution grid).
  - German PRDx whitelist/blacklist; postprocessing runners.
- `docs/topology_optimizer/intro.md` (includes the package README), `structure.md`, `deepdive.md` (marked WIP on the "genetic algorithm"): two-stage DC→AC pipeline, quality-diversity, Kafka topics.
- `docs/topology_optimizer/metrics.md`: authoritative definitions of every metric (matrix, operation, other), including exponential overload α=1.5, n0_n1_delta, cross-coupler flow, busbar-outage penalty and islands, AC-only va-diff and voltage-jump metrics, and non-converging loadflows.
- `docs/topology_optimizer/ac/select_strategy.md`: median, dominator and discriminator filters over repertoire heatmaps.
- `docs/topology_optimizer/ac/early_stopping.md`: worst-k contingency screening (union of AC-baseline and DC worst-k), acceptance thresholds, reuse of results.
- `docs/topology_optimizer/ac/ac_loop.md`: survivor batching, timeout flush, runtime limit.
- `docs/contingency_analysis/cascade.md`: cascade simulation.
  - Triggers: current overload (loading = i/i_max) and distance protection.
  - Three nested polygon zones (DANGER/ALARM/WARNING); factor divides measured |R|; base-case screening.
  - SpPS (Special Protection Schemes) iteration; `sw_characteristics` relay table (angle, r_i, r_v, x_v).
- `docs/contingency_analysis/propagation.md`: ToOp busbar-section expansion vs PowSyBl native `contingencyPropagation` (breaker-based fault isolation, currently disabled); pandapower outage grouping by connected components.
- `docs/contingency_analysis/result_filtering.md`, `progress.md`: result filtering (loading and voltage-band thresholds); Ray progress callbacks.
- `docs/interfaces/asset_topology.md`: node-breaker ↔ bus-branch mapping; MasterBusGroup/RuntimeBusGroup; Busbar, BusbarCoupler, SwitchableAsset; AssetBay (disconnector–breaker–selector disconnectors); AssetSetpoint (PST/HVDC).
- `docs/importer/network_graph/index.md`: NetworkGraphData schemas (busbar/node, DISCONNECTOR/BREAKER, branch types for powsybl and pandapower, helper branches).
  - Weighted shortest paths with cutoffs to find bays, couplers and zero-impedance connections.
  - No star-equivalent needed for trafo3w.
- `docs/importer/pypowsybl/merge_ucte_with_cgmes.md`: border lines, area codes, boundary voltage levels → tie lines, merge, AC loadflow.
- `docs/importer/contingency_from_power_factory/index.md`: PowerFactory contingency import with pandera validation.
- `docs/importer/exporter/index.md`: UCTE export (RES/RAS, PRDx) and PowerFactory DGS export.
- `docs/importer/worker/worker.md`: importer Kafka worker.
- `docs/usage.md`, `docs/quickstart.md`, `docs/benchmark.md`: Kafka workflow; Hydra multirun and parameter sweeps.
- `docs/wall_of_shame.md`: known limitations.
  - Islanding branches are excluded from N-1 because LODF cannot handle them.
  - AC runs on the reduced outage set.
  - Topologies that improve convergence but worsen overload are rejected.
  - PowSyBl `NO_CALCULATION` is treated as a failure.
- `docs/architecture/` (LikeC4 `.c4` files): C4 model; Kafka topics `importer_commands`, `importer_results`, `importer_heartbeat`, `commands`, `results`, `heartbeat`; three filesystems/stores.

---

## 3. Concept inventory by knowledge domain

The dc_solver source directory below is `.../packages/dc_solver_pkg/src/toop_engine_dc_solver`. Other packages follow the same pattern, `.../packages/<pkg>/src/toop_engine_<name>`.

### 3.1 Mathematics: linear algebra

| Concept | How the code uses it | Key files | Level |
|---|---|---|---|
| Vectors and matrices, matrix-vector product, outer product | Flows `F = PTDF @ P` (einsum `"bij,btj->bti"`). Bus splits are rank-1 updates `PTDF + outer(u, v)`; disconnection sets and multi-outages are rank-k updates. | `jax/compute_batch.py:463`, `jax/bsdf.py:403`, `jax/disconnections.py:61` | B |
| Directed incidence / connectivity matrix A (branch×node, +1/−1) | `get_connectivity_matrix` builds `Cft = Cf - Ct` as sparse CSC | `preprocess/helpers/ptdf.py` | B |
| Weighted Laplacian `B_bus = Aᵀ diag(b) A`; branch-node matrix `B_f = diag(b) A` | Nodal susceptance matrix of the DC power flow ("makeBdc") | `preprocess/helpers/ptdf.py:get_susceptance_matrices` | I |
| Singular matrices, reference / slack elimination (reduce by a row and column) | PTDF is solved with the slack column removed and the reference bus fixed. NaNs mean the grid is disconnected, i.e. B is singular. | `preprocess/helpers/ptdf.py:compute_ptdf` | I |
| Solving linear systems, matrix inverse, multiple right-hand sides | `spsolve(B_red.T, B_f.T).T` gives the whole PTDF at once | same | I |
| Rank-1 updates / Sherman–Morrison | LODF-based PTDF update after a disconnection, and BSDF update after a bus split, are both rank-1 corrections with a scalar denominator. Denominator 0 means islanding. | `jax/lodf.py`, `jax/disconnections.py`, `jax/bsdf.py` | A |
| Low-rank updates / Woodbury identity | MODF: solve `(I − H_OO) X = H_MO` (k×k). Branch-parameter change: `(I + D_α H_OO) C = D_α PTDF_O`, then `PTDF += (−H + E) C`. The identity is not named in the code, but the algebra is Woodbury. | `jax/multi_outages.py:build_modf_matrix`, `update_ptdf_with_modf`; `jax/branch_parameter_changes.py` | A |
| Determinants, 2×2 / 3×3 closed-form inverse (adjugate/Cramer) | Unrolled solves for 2- and 3-branch trafo3w multi-outages, with a determinant check | `jax/unrolled_linalg.py` | B–I |
| Superposition / linearity | Injection outage `F' = F + PTDF[:,node]·ΔP`; reassignment ΔP on bus A/B; `Flow = PTDF·P + PSDF·α` | `jax/contingency_analysis.py:calc_injection_outage`, `jax/injections.py:get_reassignment_deltap`, `preprocess/helpers/psdf.py` | B |
| Block / augmented matrices | PTDF extended with "bus B" columns (copies) for each relevant substation; PSDF appended as extra pseudo-node columns | `preprocess/helpers/ptdf.py:get_extended_ptdf`, `preprocess/preprocess.py:combine_phaseshift_and_injection`, `add_bus_b_columns_to_ptdf` | I |
| Tensor index notation (einsum), batching axes | Everywhere in the solver (`io,to->ti`, `bij,btj->bti`) | `jax/multi_outages.py`, `jax/compute_batch.py` | I |
| Monoid / associativity (a design comment) | Comment on whether BSDF composition is a monoid, which would allow `reduce` | `jax/bsdf.py:535` | A (optional) |

### 3.2 Mathematics: numerical linear algebra and numerics

| Concept | Use | Files | Level |
|---|---|---|---|
| Sparse matrix storage (CSC/CSR, COO construction) | B matrices, nodal injection aggregation | `preprocess/helpers/ptdf.py`, `preprocess/helpers/injection_topology.py` | I |
| Sparse direct solvers (SuperLU via `scipy.sparse.linalg.spsolve`; UMFPACK option in pandapower) | PTDF computation; pandapower parameter `use_umfpack` | `preprocess/helpers/ptdf.py`; `grid_helpers_pkg/.../pandapower/loadflow_parameters.py` | I |
| LU factorization with pivoting; determinant from the LU diagonal | General k×k MODF solves in JAX (`jax.scipy.linalg.lu_factor/lu_solve`) | `jax/unrolled_linalg.py:solve_and_check_det` | I |
| Conditioning, tolerances, near-singularity detection | Thresholds: `|denom| > 1e-11` (LODF), `≥ 1e-5` (BSDF), `|det| > 1e-10`; NaN/finite checks | `jax/lodf.py`, `jax/bsdf.py`, `jax/unrolled_linalg.py` | I |
| Floating-point precision (float32 vs float64) | `jax.config.update("jax_enable_x64", True)` is recommended because JAX defaults to 32-bit | `docs/dc_solver/quickstart.md` | B–I |
| Newton–Raphson iteration for nonlinear systems (convergence, mismatch tolerance, iteration caps, initialization) | AC power flow in pandapower (`algorithm: "nr"`, `max_iteration`, `tolerance_mva: 1e-8`) and OpenLoadFlow (`maxNewtonRaphsonIterations: 25`, `maxActivePowerMismatch`); warm start `init="results"` | `grid_helpers_pkg/.../pandapower/loadflow_parameters.py`, `grid_helpers_pkg/.../powsybl/loadflow_parameters.py`, `contingency_analysis_pkg/.../pandapower/outage_power_flow.py` | I–A |
| Padded arrays and sentinel indices (fixed shapes for GPU) | `int_max` padding with `.at[].get(mode="fill")` so variable-size data fits a static shape | `jax/bsdf.py`, `jax/lodf.py`, `jax/types.py` | I |

### 3.3 Mathematics: graph theory and algorithms

| Concept | Use | Files | Level |
|---|---|---|---|
| Graphs, multigraphs (parallel branches), directed vs undirected | Grid as a (multi)graph keyed by branch index | `preprocess/helpers/find_bridges.py` | B |
| Connected components / islanding | Grid-split detection; outage groups (pandapower); 2-component coupler partitions; SciPy csgraph; union-find bus lookup | `preprocess/preprocess_switching.py`, `grid_helpers_pkg/.../pandapower/outage_group.py`, `slack_allocation.py`, `bus_lookup.py` | B–I |
| Bridges (Tarjan, `nx.bridges`) | Branches whose outage islands the grid are removed from the N-1 set; "mainland" side; N-2-safe disconnectables found by counting bridges after each removal | `preprocess/helpers/find_bridges.py`, `preprocess/preprocess.py:compute_bridging_branches`, `filter_disconnectable_branches_nminus2` | I |
| Articulation points (cut vertices) | Busbars whose outage splits a station or grid are excluded from busbar outages | `preprocess/preprocess_bb_outage.py:get_articulation_nodes`, `preprocess/network_data.py:839` | I |
| Weighted shortest paths (Dijkstra, cutoff, custom multi-weight functions) | Node-breaker parsing: find asset bays, couplers, busbar connections | `grid_helpers_pkg/.../network_graph/filter_strategy/*.py`, `network_graph_helper_functions.py` | I |
| k-hop neighborhoods | Node-dimension reduction keeps nodes within 2 branches of relevant elements | `preprocess/helpers/reduce_node_dimension.py` | B |
| Exhaustive enumeration over subsets (`itertools.product`) plus a component test | All coupler open/closed states giving exactly two electrical buses (≤ 20 couplers) | `preprocess/preprocess_switching.py:make_separation_set` | I |
| Graph partitioning (Kernighan–Lin bisection) | Only for generating synthetic region masks in example grids | `dc_solver/example_grids.py:987` | A (optional) |
| Hamming distance between binary vectors | Switching distance; configuration deduplication | `preprocess/helpers/switching_distance.py` | B |
| Combinatorics of splits (2^(n−1) bus assignments, clipping) | Action enumeration per station (`action_set_clip = 2**23`); exhaustive bruteforce generator | `preprocess/action_set.py`, `topology_optimizer_pkg/.../dc_bruteforce/generator.py` | B |

### 3.4 Mathematics: complex numbers, trigonometry, geometry

| Concept | Use | Files | Level |
|---|---|---|---|
| Phasors, complex impedance Z = R + jX, polar ↔ rectangular | Relay impedance: `V_phase = V_LL/√3`, `|Z| = V/I`, `φ = atan2(Q, P)`, R and X from φ | `contingency_analysis_pkg/.../pandapower/cascade/detection/switch_preparation.py:get_complex_impedance` | B–I |
| Apparent power magnitude `S = √(P² + Q²)`; three-phase current `I = S/(√3·V)` | Switch current results; P-to-I conversion; limits `P_limit = √3·V·I_perm` | `contingency_analysis_pkg/.../results/switch_results.py:590`, `grid_helpers_pkg/.../powsybl/powsybl_helpers.py:98-165`, `dc_solver_pkg/.../preprocess/powsybl/powsybl_helpers.py:get_p_max` | B |
| Angle arithmetic, `atan2`, degrees ↔ radians | AC result post-processing computes a PST angle from tap position, step percent and step degree with `atan2` (the DC solver instead reads angles from `pst_tap_values`); PSDF in MW/° (× base_MVA·π/180); critical angle difference 20° | `contingency_analysis_pkg/.../results/va_diff_results.py:540-600`, `preprocess/helpers/psdf.py` | B |
| Computational geometry: polygon construction, point-in-polygon (shapely) | Distance-protection quadrilateral zones in the R-X plane | `contingency_analysis_pkg/.../cascade/detection/distance_protection.py` | I |

### 3.5 Mathematics: probability and statistics (light)

| Concept | Use | Files | Level |
|---|---|---|---|
| Discrete distributions: categorical sampling with probabilities, Poisson, Normal | Number of substations mutated ~ Poisson(λ=2); PST tap mutation draws an integer step from a discretized Gaussian on ±⌈4σ⌉ (defaults σ=1, probability 0.2, reset probability 0.0 in `BatchedMEParameters`); the Poisson draw is clipped to [1, max_num_splits]; uniform sampling from the repertoire | `topology_optimizer_pkg/.../dc/genetic_functions/mutation/mutate.py`, `mutate_nodal_inj.py`, `dc/repertoire/discrete_me_repertoire.py:sample` | B |
| Pseudo-random number generators, seeds, key splitting (JAX PRNG) | Reproducible mutation and crossover | same; `dc/genetic_functions/crossover.py` | I |
| Order statistics: max, median, top-k | N-1 worst case per branch; median metrics; median filter; `jax.lax.top_k` worst-k contingencies | `jax/aggregate_results.py`, `ac/select_strategy.py` | B |

### 3.6 Electrical engineering fundamentals

| Concept | Use | Files | Level |
|---|---|---|---|
| Kirchhoff's current and voltage laws, Ohm's law | DC power flow derivation; cross-coupler flow = KCL imbalance at bus A | `jax/cross_coupler_flow.py:compute_cross_coupler_flow_single` | B |
| Sinusoidal steady state, active/reactive/apparent power, power factor | AC results (p, q, i, vm, va); `dc_power_factor = 1.0` when converting current limits to MW | `grid_helpers_pkg/.../powsybl/loadflow_parameters.py`, loadflow result schemas in `interfaces_pkg/.../loadflow_results*.py` | B–I |
| Three-phase systems (line-to-line vs phase voltage, √3 factors) | Current and limit conversions; relay phase voltage | see §3.4 | B |
| Per-unit system, base MVA | PowSyBl `net_pu.per_unit = True`; susceptances in p.u.; PSDF scaled by base_mva; pandapower ppc internal format | `preprocess/powsybl/powsybl_helpers.py:get_network_as_pu`, `powsybl_backend.py:get_base_mva` | I |
| Series reactance and susceptance (b = 1/x; DC neglects r) | `get_susceptances = 1/x`; pandapower `calc_b_from_branch` | `preprocess/powsybl/powsybl_backend.py:431`, `preprocess/pandapower/pandapower_backend.py:503` | B |
| Transformers: ratio (rho), tap changers, T vs π equivalent models | DC susceptance uses `x/rho` (`dc_use_transformer_ratio=True`); pandapower `trafo_model: "t"`; low-voltage-side limits | `preprocess/powsybl/powsybl_helpers.py:get_trafos`, `grid_helpers_pkg/.../pandapower/loadflow_parameters.py` | I |
| Phase-shifting transformers (PST): symmetric/asymmetric, linear vs non-linear tap tables, tap steps (alpha, rho, x) | Taps optimized by index with per-tap angle and susceptance tables (`pst_tap_values`, `pst_tap_susceptance_values`); `_is_linear_pst_step_table` (rho, x, r, g, b constant) is used only to group parallel PSTs into buckets, and the DC `pst_linear` flag comes from `get_linear_pst` (x constant across taps) | `preprocess/powsybl/powsybl_backend.py:493-561`, `preprocess/powsybl/powsybl_helpers.py:_is_linear_pst_step_table`, `_identify_pst_buckets`, `jax/pst.py` | I–A |
| Three-winding transformers (star equivalent with auxiliary bus) | pandapower models each as 3 branches plus an aux bus; outage = 2- or 3-branch multi-outage | `preprocess/pandapower/pandapower_backend.py:708`, `jax/multi_outages.py`, `preprocess/preprocess.py:convert_multi_outages` | I |
| HVDC (LCC and VSC converter stations), dclines | Modeled as P injections; `HVDCRange` setpoint placeholder | `preprocess/powsybl/powsybl_backend.py:_get_hvdc_lcc`, `_get_hvdc_vsc`; `interfaces_pkg/.../stored_action_set.py:HVDCRange` | I |
| Generators, loads, batteries, boundary lines, HVDC converter stations; shunts and SVCs; constant-Z loads | PowSyBl injection table holds generators, loads, boundary lines, batteries and LCC/VSC converters (the backend asserts there are no shunt compensators with active power and no three-winding transformers; those are handled in the pandapower backend); example grid "battery, HVDC, SVC, 3w trafo"; `modify_constan_z_load` | `grid_helpers_pkg/.../powsybl/example_grids.py:create_complex_grid_battery_hvdc_svc_3w_trafo`, `importer_pkg/.../pandapower_import/preprocessing.py` | B–I |
| Network equivalents: Ward / extended Ward (xward), boundary (dangling) lines, X-nodes, tie lines (x = x₁ + x₂) | Aux buses for xward; boundary lines as injections; tie lines merge two dangling halves | `preprocess/pandapower/pandapower_backend.py:210-230`, `preprocess/powsybl/powsybl_helpers.py:get_tie_lines`, `importer_pkg/.../merge_ucte_cgmes/` | I–A |
| Protection: circuit breakers vs disconnectors (load-break), relays, distance (impedance) protection, zones, overcurrent tripping | Cascade simulation; contingency propagation via breakers | `contingency_analysis_pkg/.../pandapower/cascade/`, `docs/contingency_analysis/cascade.md`, `propagation.md` | I–A |

### 3.7 Power systems: power flow

| Concept | Use | Files | Level |
|---|---|---|---|
| Bus-branch model, nodal injections, slack/reference bus, PV/PQ buses | Backend interface exposes nodes, branches and injections; single slack for PTDF (distributed slack "cannot" be used with BSDF) | `interfaces_pkg/.../backend.py`, `preprocess/helpers/ptdf.py:43` | B–I |
| **DC power flow approximation** (flat voltage, small angles, r ≪ x, P = B·θ) | Whole GPU solver; the pandapower `rundcpp` and PowSyBl `run_dc` paths | `preprocess/helpers/ptdf.py`, `jax/*` | I |
| **AC power flow** (nonlinear power balance equations, Newton–Raphson, reactive limits, voltage control, convergence) | AC validation and N-1 in pandapower `runpp` and PowSyBl OpenLoadFlow | `contingency_analysis_pkg/.../pandapower/contingency_analysis_pandapower.py:852`, `pypowsybl/contingency_analysis_powsybl.py:189`, `grid_helpers_pkg/.../*/loadflow_parameters.py` | I–A |
| Distributed slack (balance proportional to Pmax or P), reference bus selection, slack per island | OpenLoadFlow `BalanceType`; slack allocation to generators per island (pandapower) | `grid_helpers_pkg/.../powsybl/loadflow_parameters.py`, `grid_helpers_pkg/.../pandapower/slack_allocation.py` | I |
| Voltage initialization modes (flat, DC values, previous values) and convergence fallbacks | Three inits are tried at import; parameters are persisted | `docs/dc_solver/loadflow_parameters.md`, `importer_pkg/.../pypowsybl_import/preprocessing.py` | I |
| AC–DC mismatch correction | `unsplit_flow = PTDF·P + ac_dc_mismatch·ac_dc_interpolation` | `jax/cross_coupler_flow.py:179-203`, `preprocess/*/..._backend.py:get_ac_dc_mismatch` | I |
| Multi-timestep ("chronics") | Arrays carry an `n_timesteps` axis; the harmonize module is disabled | `jax/types.py`, `preprocess/harmonize.py`, `preprocess/pandapower/pandapower_backend.py` (dcline_p chronics) | B |

### 3.8 Power systems: linear sensitivity (distribution) factors, the core of the solver

| Factor | Formula / algorithm in the code | Files | Level |
|---|---|---|---|
| **PTDF** (branch × node) | `PTDF[:, noslack] = spsolve(B_bus[noslack, noref]ᵀ, B_f[:, noref]ᵀ)ᵀ` | `preprocess/helpers/ptdf.py:compute_ptdf` | I |
| **PSDF** (phase shift) | `PSDF = −b_pst·(PTDF[:, f] − PTDF[:, t])`, plus `b_pst` on the PST's own row; scaled by `base_mva * (np.pi/180)` in `psdf.py` (MW/°). Stacked onto the PTDF so one matmul gives flows. | `preprocess/helpers/psdf.py`, `preprocess/preprocess.py:combine_phaseshift_and_injection` | I |
| **LODF** (single outage) | `LODF_{a,b} = (PTDF_{a,f_b} − PTDF_{a,t_b}) / (1 − (PTDF_{b,f_b} − PTDF_{b,t_b}))`; own row = −1; failure when the denominator ≈ 0 (bridge); `n_1 = n_0 + LODF·n_0[b]` | `jax/lodf.py`, `jax/contingency_analysis.py:calc_n_1_matrix` | I |
| PTDF update after topological disconnections | In the batch path, `apply_disconnections` applies all disconnections of a topology as one joint MODF update (`build_modf_matrix` + `update_ptdf_with_modf`); a failure marks the topology unsuccessful. The single-outage form `PTDF' = PTDF + outer(LODF, PTDF[b]); PTDF'[b] = 0` (`apply_single_disconnection_lodf`) is used only in tests | `jax/disconnections.py`, `jax/multi_outages.py` | I–A |
| **MODF** (multi-outage, generalized LODF) | `denom = I − H_OO`, `nom = H_MO` (own rows set to −denom), `MODF = solve(denomᵀ, nomᵀ)ᵀ`; flows `F0 + MODF·F0[O]`; PTDF update uses "equation 7" of L6 | `jax/multi_outages.py` | A |
| **BSDF** (bus split) | `ptdf_th_sw` = Σ PTDF rows of branches entering bus A − Σ leaving (slack handled); `denom = Σ b_A − g_sw`; `nom = Σ b·PEDF[other ends]` ± b on own rows; `PTDF' = PTDF + outer(bsdf, ptdf_th_sw)`; from/to node vectors are rewired to bus B | `jax/bsdf.py:calc_bsdf`, `_apply_bus_split`, `compute_bus_splits` | A |
| PEDF (theoretical injection to bus B) | `pedf = PTDF − PTDF[:, i_stat]` inside BSDF | `jax/bsdf.py` | A |
| Susceptance-change (PST tap) low-rank update | `α = Δb/b`, `(I + D_α H_OO)·C = D_α·PTDF_O`, then `PTDF += (E − H)·C` | `jax/branch_parameter_changes.py` | A |
| Injection outages and reassignment | ΔP at a node × PTDF column; bus A/B deltas | `jax/contingency_analysis.py`, `jax/injections.py` | I |
| Order of composition | PTDF side: bus splits (BSDF) → disconnections (joint MODF) → PST susceptance update (Woodbury) → LODF and MODF contingency matrices. Flow side: nodal injections → cross-coupler flows → N-0 update after disconnections → PST angles (N-0 recomputed) → N-1 flows, injection outages and busbar outages (MODF + ΔP) | `jax/compute_batch.py:compute_bsdf_lodf_static_flows`, `jax/topology_looper.py` docstring | A |
| NumPy reference implementations (for teaching) | Plain-NumPy LODF/BSDF, also used for the action-set BSDF pre-filter | `packages/dc_solver_pkg/tests/numpy_reference.py` | I |

### 3.9 Power systems: security, contingency analysis, operations

| Concept | Use | Files | Level |
|---|---|---|---|
| N-0, N-1, N-2 security criterion; monitored vs outaged elements; contingency lists | `Nminus1Definition` (Contingency, GridElement, MonitoredElement); N-2-safe disconnectables; `n_2_penalty` | `interfaces_pkg/.../nminus1_definition.py`, `preprocess/helpers/find_bridges.py` | B–I |
| Contingency types: branch, injection (generator/load), busbar, trafo3w multi-outage, exceptional/multi | Branch and injection outages, busbar outages (relevant and non-relevant), multi-outages | `jax/contingency_analysis.py`, `jax/busbar_outage.py`, `preprocess/preprocess_bb_outage.py` | I |
| Security analysis engines | PowSyBl `security.create_analysis().run_ac/run_dc`; pandapower sequential or Ray-parallel | `contingency_analysis_pkg/.../pypowsybl/contingency_analysis_powsybl.py`, `.../pandapower/contingency_analysis_pandapower.py` | I |
| Thermal ratings: permanent limit (PATL), N-1 / temporary limits, current → MW conversion, loading = I/I_max | `get_p_max` probes `permanent_limit` and "N-1" limits; loadflow-based artificial limits for border lines and DSO transformers (`LimitAdjustmentParameters`) | `preprocess/powsybl/powsybl_helpers.py:149-198`, `importer_pkg/.../pypowsybl_import/loadflow_based_current_limits.py`, `interfaces_pkg/.../preprocess_commands.py:47` | I |
| Overload metrics | Overload energy `Σ_t max_contingency clip(|F| − Fmax, 0)` (weighted); "exponential" `((|F|/Fmax)^α − 1)·Fmax` with α=1.5 (actually a power law); critical branch count; transport; underload; cumulative overload; median/max flow | `jax/aggregate_results.py`, `docs/topology_optimizer/metrics.md` | B–I |
| Double limits (operational banding 0.9/1.0) | `compute_double_limits` | `jax/aggregate_results.py:999` | B |
| N0–N1 max-diff (limit power pushed to DSO after a contingency) | `n0_n1_delta` penalty | `jax/aggregate_results.py:407-518`, `docs/dc_solver/quickstart.md` | I |
| Voltage security metrics (AC only) | Voltage jump N-0→N-1 > 5%; voltage-angle difference across breakers at contingency ends > 20° (reclosing/synchro-check motivation, not stated in the repo); voltage bands in result filters | `contingency_analysis_pkg/.../ac_loadflow_service/compute_metrics.py`, `.../pandapower_helpers/va_diff_info.py`, `results/va_diff_results.py` | I |
| Congestion management; redispatch as the costly alternative; non-costly remedial actions (topology, PST); preventive vs curative | Motivation (README); OpenRAO "preventive-actions-list" export | `README.md`, `topology_optimizer_pkg/.../ac/summary.py`, `notebooks/openrao.ipynb` | B–I |
| Cascading failures; Special Protection Schemes (SpPS rule engine: conditions/actions, ALL/ANY, iterate power flow) | Optional pandapower cascade loop with depth limit and minimum island size | `contingency_analysis_pkg/.../pandapower/spps/engine.py`, `.../cascade/simulation/simulator.py`, `interfaces_pkg/.../nminus1_definition.py:SppsRule` | A |
| Distance protection: R-X quadrilateral (angle, r_i, r_v, x_v), severity zones, per-relay factors | Trip detection in cascades | `.../cascade/detection/distance_protection.py`, `docs/contingency_analysis/cascade.md` | A |
| Contingency propagation / fault isolation | Busbar-section expansion; PowSyBl breaker-based propagation (disabled); connected-component outage groups | `docs/contingency_analysis/propagation.md`, `grid_helpers_pkg/.../pandapower/outage_group.py` | I |
| TSO operational context: control / view / N-1 areas, UCTE area codes and ISO 3166 regions, cutoff voltage (220 kV default), TSO-TSO border lines, DSO border transformers, German PRDx and DACF processes, critical-branch whitelist/blacklist, RES/RAS | Importer masks and area settings | `importer_pkg/.../pypowsybl_import/powsybl_masks.py`, `dacf_whitelists.py`, `network_analysis.py:apply_cb_lists`, `interfaces_pkg/.../preprocess_commands.py` | I |
| Islanding penalties and grid-split handling | `bb_outage_grid_splits`, `bb_outage_more_islands_penalty`; bridges excluded | `jax/busbar_outage.py`, `dc_params.py` | I |

### 3.10 Power systems: substation topology, grid models and data formats

| Concept | Use | Files | Level |
|---|---|---|---|
| Node-breaker vs bus-breaker vs bus-branch models | Importer converts node-breaker → bus-branch; asset topology maps both ways | `docs/interfaces/asset_topology.md`, `importer_pkg/.../pandapower_import/preprocessing.py`, `grid_helpers_pkg/.../powsybl/powsybl_station_to_graph.py` | I |
| Substation layout: busbars and busbar sections, busbar couplers, cross couplers, bays / asset bays, breakers, selector disconnectors, double busbar, zero-impedance connections, fusing closed switches | NetworkGraph filter strategy; `pandapower_toolset_node_breaker` (ASCII diagram in the docstring) | `grid_helpers_pkg/.../network_graph/`, `importer_pkg/.../pandapower_import/pandapower_toolset_node_breaker.py` | I |
| Topological remedial actions: busbar splitting, reassignment, line switching, PST taps | ActionSet; topo-vect bus A/B (grid2op); a topology is actions + disconnections + pst_setpoints | `interfaces_pkg/.../stored_action_set.py`, `topology_optimizer_pkg/.../interfaces/messages/results.py:Topology` | B–I |
| Electrical vs physical switching; station realization heuristics; switching distance (reassignment + coupler distance) | Separation sets and Hamming distance; `realise_ba_to_physical_topo_per_station_jax` | `preprocess/preprocess_switching.py`, `preprocess/helpers/switching_distance.py`, `postprocess/realize_assignment.py`, `preprocess/preprocess_station_realisations.py` | I–A |
| Master vs runtime vs simplified asset topology; bus_group_id identity | Structural vs electrical grouping | `interfaces_pkg/.../asset_topology/`, `preprocess/simplify_topology.py` | I |
| Single-line diagrams (SLD), network area diagrams (NAD) | Before/after SLDs of split stations; custom PowSyBl SLD styling | `grid_helpers_pkg/.../powsybl/single_line_diagram/`, `grid_helpers_pkg/.../powsybl/loadflow_parameters.py:SDL_PARAM`, `notebooks/example2_small_grid_toop.ipynb` | B |
| Formats: UCTE-DEF (.uct fixed-width), CGMES/CIM (.zip), PowSyBl IIDM/XIIDM, pandapower JSON, MATPOWER .mat (matpowercaseframes), PowerFactory DGS export and contingency import, OpenRAO CRAC/GLSK (notebook) and ORAO JSON summary | Import, export, merge | `importer_pkg/.../ucte_toolset/ucte_io.py`, `.../pypowsybl_import/cgmes/`, `.../exporter/`, `dc_solver_pkg/.../export/asset_topology_to_dgs.py`, `dgs_v7_definitions.py`, `importer_pkg/.../contingency_from_power_factory/` | I |
| UCTE + CGMES merging (boundary voltage levels → tie lines) | Multi-TSO model merge | `importer_pkg/.../pypowsybl_import/merge_ucte_cgmes/`, `docs/importer/pypowsybl/merge_ucte_with_cgmes.md` | A |
| Network reduction (voltage levels, depth range), low-impedance lines, removing branches across switches | Import cleanup | `importer_pkg/.../pypowsybl_import/network_reduction.py`, `network_analysis.py` | I |
| Standard test cases: IEEE 14/30/57/300; PEGASE-like case1354 / case9241; pandapower Oberrhein; CIGRE MV | Examples and tests | `dc_solver_pkg/.../example_grids.py`, `grid_helpers_pkg/.../*/example_grids.py` | B |

### 3.11 Optimization

| Concept | Use | Files | Level |
|---|---|---|---|
| Combinatorial / discrete optimization; search-space size | Topologies = choice of ≤ `max_num_splits` station actions (one per substation) plus ≤ `max_num_disconnections` plus PST taps | `dc_params.py:LoadflowSolverParameters`, `jax/topology_computations.py` | I |
| Objective design: weighted-sum scalarization, penalties as soft constraints | `fitness = Σ −w·metric`; infeasible (split grid) scored as `inf`; penalties n0_n1_delta, cross_coupler, busbar outage | `topology_optimizer_pkg/.../dc/genetic_functions/scoring_functions.py:116,266` | B–I |
| Evolutionary algorithms: genotype, mutation (add/change/remove split or disconnection, PST Gaussian), crossover (unique sampling), random restarts (`random_topo_prob`), deduplication, elitism | Discrete GA operators in JAX | `dc/genetic_functions/genotype.py`, `mutation/*.py`, `crossover.py`, `dc/ga_helpers.py` | I |
| **Quality-diversity / MAP-Elites**: descriptors (split_subs, switching_distance, disconnected_branches), grid cells (`ravel_multi_index`), per-cell elite, `cell_depth`, emitters, `segment_max` insertion | `DiscreteMapElites`, `DiscreteMapElitesRepertoire` (adapted from QDax) | `dc/repertoire/discrete_map_elites.py`, `discrete_me_repertoire.py`, `dc/repertoire/plotting.py` | A |
| Multi-objective concepts: Pareto optimality, dominance | README claims a Pareto front; the actual dominance heuristic is `get_dominator_mask` over discrete target metrics; median and discriminator filters | `README.md:101`, `ac/select_strategy.py` | I |
| Exhaustive (bruteforce) search with lazy enumeration | `dc_bruteforce` optimizer | `topology_optimizer_pkg/.../dc_bruteforce/generator.py`, `optimizer.py` | B |
| Two-fidelity pipeline: surrogate/linear screening (DC) then high-fidelity validation (AC); early rejection on worst-k subset; acceptance thresholds (convergence ≤1.0×, overload ≤0.95×, critical branches ≤1.1×, voltage jump and va-diff ≤1.1×) | AC worker and scoring | `topology_optimizer_pkg/.../ac/scoring_functions.py:evaluate_acceptance`, `optimizer/interfaces/messages/ac_params.py`, `docs/topology_optimizer/ac/*.md` | I |
| Explore/exploit selection | `select_strategy` and interest scorer for pulling DC candidates | `ac/select_strategy.py`, `ac/evolution_functions.py` | I |
| Hyperparameter search | Hydra multirun sweeps; optuna and ray[tune] in dependencies | `docs/benchmark.md`, `toop-engine-benchmark/`, `topology_optimizer_pkg/pyproject.toml` | I |
| Related tool: remedial action optimization (OpenRAO: CRAC, GLSK) | Export format and stub notebook only | `notebooks/openrao.ipynb`, `ac/summary.py` | A (context) |

### 3.12 Computing: JAX / GPU and numerical Python

| Concept | Use (counts are non-test source files) | Files | Conceptual weight |
|---|---|---|---|
| Functional programming, immutability, pure functions | StaticInformation updated via `replace`; notebook 1 explains this explicitly | `notebooks/example1_dc_loadflow_example.ipynb`, `jax/types.py` | High (B–I) |
| `jax.jit`, tracing, static vs dynamic arguments, recompilation | `static_argnames`; SolverConfig is static, DynamicInformation is traced; `HashableArrayWrapper` avoids recompiles | `jax/topology_looper.py:491`, `jax/utils.py`, `jax/types.py` | High (I) |
| `jax.vmap` (nested batching over topologies, injections, outages) | 19 files | `jax/compute_batch.py`, `jax/lodf.py` | High (I) |
| Control flow primitives: `lax.scan`, `fori_loop`, `cond`, `switch`, `dynamic_slice`, `top_k`, `ops.segment_max` | Batching loops; top-k; repertoire insertion | `jax/topology_looper.py:584`, `jax/injections.py:134`, `jax/cross_coupler_flow.py:322`, `discrete_me_repertoire.py` | High (I) |
| `jax.pmap` multi-device, `donate_argnums`, `jax.devices`, `default_device(cpu)` | Distributed solver and GA epoch | `jax/topology_looper.py:446`, `dc/worker/optimizer.py:334`, `dc/genetic_functions/initialization.py:310` | Medium (A) |
| Pytrees (`jax.tree_util`), equinox Modules, jax_dataclasses `Static` | All solver and GA state | `jax/types.py`, `dc/genetic_functions/genotype.py` | High (I) |
| Shape-typed arrays and runtime checking (jaxtyping + beartype, chex in tests) | Every signature, e.g. `Float[Array, " n_branches n_bus"]` | throughout | Medium (B) |
| Fixed-shape padding, GPU memory, batch sizes, OOM tuning | `batch_size_bsdf`, `batch_size_injection`, `max_num_splits` | `jax/batching.py`, `packages/dc_solver_pkg/README.md` (cost of BSDF and LODF slots) | Medium (I) |
| XLA environment flags, CUDA_VISIBLE_DEVICES, float64 | `set_environment_variables` | `topology_optimizer_pkg/.../benchmark/benchmark_utils.py` | Low–Medium |
| NumPy / SciPy sparse, NetworkX, shapely, numba (pandapower acceleration) | Preprocessing and CPU paths | as above | Medium |
| Not used: `jax.grad`, optax, evosax, lineax (commented out), sharding / `shard_map` | n/a | `jax/multi_outages.py:29` | n/a |

### 3.13 Computing: data tooling, distributed systems and messaging

| Concept | Use | Files | Weight |
|---|---|---|---|
| Tabular data: pandas and polars (lazy frames), pyarrow | Grid element tables; loadflow results; polars is faster for large N-1 | `interfaces_pkg/.../loadflow_results_polars.py`, `contingency_analysis_pkg/.../pypowsybl/powsybl_helpers_polars.py` | Medium |
| Schema validation: pandera (DataFrames), pydantic (models, validators, discriminated unions) | 58 and 50 files respectively | `interfaces_pkg/.../messages/`, `dc_params.py` validators | Medium |
| HDF5 (h5py) binary arrays; JSON; `.npy` masks; filesystem abstraction (fsspec DirFileSystem) | `static_information.hdf5`, `action_set_diffs.hdf5`, masks | `jax/inputs.py`, `interfaces_pkg/.../folder_structure.py` | Medium |
| Message-driven architecture: Kafka (confluent-kafka) topics, commands, results, heartbeats; protobuf wrapper | DC and AC workers run concurrently | `topology_optimizer_pkg/.../dc/worker/worker.py`, `ac/worker.py`, `interfaces_pkg/.../messages/protobuf_schema/message_wrapper.proto`, `docs/architecture/model/02-messaging.c4` | Medium (I) |
| Parallelism on CPU: Ray remote tasks and queues (pandapower N-1), ThreadPoolExecutor (AC runners), multiprocessing | Parallel contingencies and topologies | `contingency_analysis_pkg/.../pandapower/contingency_analysis_pandapower.py:755`, `topology_optimizer_pkg/.../ac/scoring_functions.py:10` | Medium |
| Databases: SQLModel/SQLAlchemy (SQLite) for the AC repertoire | Storing and querying topologies | `topology_optimizer_pkg/.../ac/storage.py`, `evolution_functions.py` | Low–Medium |
| Experiment tooling: tensorboardX, tyro CLIs, Hydra configs, structlog | Logging and CLI | `dc/main.py`, `benchmark_utils.py` | Low |
| Engineering: uv, Docker / devcontainer, pytest (xdist), MkDocs + mkdocstrings, LikeC4/C4 model, ruff | Development workflow | `pyproject.toml`, `docs/architecture/` | Low (for theory) |
| External power-system libraries: pandapower (PYPOWER ppc internals), pypowsybl (Java PowSyBl via C++ bindings), OpenLoadFlow | The backends | `preprocess/*_backend.py` | High (practical) |

---

## 4. Notebooks: what each teaches and which concepts it requires

Folder: ``../ToOp/notebooks/``

1. **`example1_dc_loadflow_example.ipynb`**: DC loadflow on IEEE-57 (PowSyBl backend).
   - Content: `load_grid` → static information (solver_config vs dynamic_information), then `run_initial_loadflow`; 72 nodes, 82 branches; `unsplit_flow` shape (timesteps × branches); functional programming note.
   - Requires: DC power flow, PTDF, float64 in JAX, immutability / pytrees.
2. **`example2_small_grid_toop.ipynb`**: full pipeline on a small node-breaker grid (`data/grid_node_breaker`).
   - Content: import → preprocess (busbar outages enabled, `bb_outage_as_nminus1=False`) → DC MAP-Elites (descriptor split_subs) → AC validation; single-line diagrams before and after the VL2 split; `res.json` `overload_energy_n_1`.
   - Requires: substation topology, busbar splitting, N-1, overload energy, MAP-Elites, AC power flow, SLDs.
3. **`example3_e2e_pipeline.ipynb`**: staged pipeline on a generated "complex grid with battery, HVDC, SVC, 3w trafo".
   - Content: CgmesImporterParameters and AreaSettings (cutoff voltage, control/view/N-1 areas); `target_metrics` weighted sum (`overload_energy_n_0` + 50·`critical_branch_count_n_0`); three descriptors (the comment "cells on the pareto front" is loose); `max_num_splits` / `max_num_disconnections` / `batch_size` (VRAM); AC validation settings (voltage jump 5%, va-diff 20°).
   - Requires: all of the above plus HVDC/3w-transformer modeling, weighted-sum objectives, GPU memory.
4. **`example_asset_topology_analysis.ipynb`**: interactive PowSyBl topology inspection (ipywidgets, pypowsybl_jupyter SLDs).
   - Content: voltage-level, bus-group and element views; master vs runtime vs simplified topology; separation-set preparation (`prepare_for_separation_set`, `StationProblems`).
   - Requires: node-breaker modeling, busbars/couplers/bays, asset topology data model.
5. **`run_ac_contingency_analysis.ipynb`**: standalone AC N-1 in pandapower (CIM import, Ray with 15 processes) and PowSyBl (CGMES).
   - Content: building `Nminus1Definition` (monitored lines, trafos, trafo3w, buses; line and trafo outages); result-count arithmetic (2 sides per branch, 3 per trafo3w); pandera validation toggle; profiling.
   - Requires: N-1 criterion, AC power flow, grid data formats, dataframes.
6. **`openrao.ipynb`**: stub, not runnable as-is.
   - Content: loads the CIGRE MV network, then a PowSyBl RAO with CRAC and GLSK files.
   - Requires: remedial action optimization context (CRAC = contingency list, remedial actions and constraints; GLSK = generation and load shift keys). These terms are not explained in the repo.
7. `notebooks/tests/test_notebooks.py` executes the notebooks.

---

## 5. Suggested dependency order of concepts (curriculum spine)

1. **Math foundations (B).** Vectors and matrices, products, outer products, solving Ax=b, determinants, inverse; complex numbers and phasors; trigonometry and atan2; Hamming distance; basic probability (categorical, Poisson, Normal); max, median, top-k.
2. **Circuits (B).** Ohm's law, KCL/KVL; nodal analysis as a matrix equation (conductance Laplacian); superposition.
3. **AC fundamentals (B–I).** Sinusoidal steady state, impedance and admittance, P/Q/S, power factor; three-phase and √3 relations; per-unit system.
4. **Power system components (I).** Lines (π model, x ≫ r), transformers (ratio, tap changers, T/π models), PSTs (angle and tap tables, linearity), three-winding transformers (star equivalent), generators, loads, shunts/SVC, batteries, HVDC LCC/VSC as injections, Ward equivalents, tie and boundary lines.
5. **Graph theory (B–I).** Graphs and multigraphs, incidence matrix, weighted Laplacian (link to step 2), connected components, bridges, articulation points, Dijkstra with cutoffs, subset enumeration and union-find.
6. **Power flow (I).** Bus types and slack; AC power flow equations; Newton–Raphson (with numerics: Jacobian, convergence, initialization); distributed slack; then the **DC approximation**, B_bus = AᵀDA, B_f = DA and the slack reduction.
7. **Numerical linear algebra (I).** Sparse storage and sparse direct solves; LU with pivoting; conditioning and tolerances; float32 vs float64.
8. **Sensitivity factors (I–A).** PTDF → PSDF → LODF (derived from PTDF) → rank-1 updates / Sherman–Morrison → MODF (multi-outage) / Woodbury → BSDF bus splits (needs substation topology from step 9) → susceptance-change updates; composition order and islanding (zero denominators ⇔ bridges).
9. **Substation topology and grid models (I).** Busbars, couplers, breakers vs disconnectors, bays; node-breaker vs bus-breaker vs bus-branch; topo-vect bus A/B; electrical vs physical switching; switching distance; asset topology. Data formats: UCTE, CGMES/CIM, XIIDM, pandapower JSON, MATPOWER, DGS. SLDs.
10. **Operational security (I).** N-0/N-1/N-2; contingency definitions (branch, injection, busbar, multi-outage); thermal limits (PATL, N-1 limits, current ↔ MW); overload metrics, double limits, N0-N1 delta; voltage-jump and angle-difference metrics; congestion management, redispatch and non-costly topological or PST remedial actions; TSO areas and processes (DACF, PRDx, border lines, DSO transformers).
11. **Protection and cascades (A, optional module).** Breakers and relays, distance protection in the R-X plane (zones, polygons), overcurrent trips, special protection schemes, contingency propagation, cascade simulation.
12. **Optimization (I–A).** Combinatorial search spaces; objectives, weighted sums and penalties; exhaustive search; evolutionary algorithms (mutation, crossover, dedup, elitism); multi-objective basics (Pareto dominance); quality-diversity and MAP-Elites (descriptors, repertoires); multi-fidelity screening (DC→AC), early rejection, acceptance thresholds; explore/exploit selection; hyperparameter sweeps; context on RAO/OpenRAO.
13. **Computing for the GPU solver (I–A).** Python typing and dataclasses; NumPy broadcasting and einsum; functional programming; JAX (jit/tracing/static args, vmap, scan/fori_loop/cond, PRNG keys, pytrees/equinox, fixed-shape padding, pmap, memory/batching); jaxtyping + beartype.
14. **Data and distributed engineering (I).** pandas/polars/pyarrow, pandera/pydantic schemas, HDF5/JSON/npy, fsspec; Kafka message-driven workers and protobuf; Ray and process/thread pools; SQL (SQLModel/SQLite); logging, tensorboard, Hydra/tyro; Docker/uv/pytest/MkDocs; C4 architecture diagrams.
15. **Capstone.** Read L1 (arXiv:2501.17529 / PowerTech DOI), L4 (BSDF), L5 (arXiv:2412.16164), L6/L7 (MODF) and L3 (arXiv:2605.10128). Then run notebooks 1 → 4 → 5 → 2 → 3 and map each stage back to `jax/compute_batch.py`, `dc/repertoire/*` and `ac/scoring_functions.py`.

Hard prerequisite edges: steps 1–2 → 5 → 6 (DC) → 8. Steps 3–4 → 6 (AC) → 10 → 11. Step 9 is needed before BSDF, busbar outages and switching distance in step 8. Steps 8 and 10 → 12. Step 13 can start in parallel after step 1, but the solver code needs step 8 first.
