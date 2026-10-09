# Chapter 28. Topology optimization in power systems

**Weeks:** 3 · **Semester:** 5 · **Prerequisites:** [Chapter 19](../part-3-power-systems/ch19-operational-security.md), [Chapter 23](ch23-optimization-modeling.md), [Chapter 27](ch27-pareto-multi-objective.md)

!!! info "Why it matters for ToOp"

    This chapter places ToOp in the literature and explains its two-stage design: a fast, approximate DC search
    followed by expensive AC validation that runs concurrently and pulls candidates from the DC repertoire
    (`optimizer/ac/`, `docs/topology_optimizer/ac/select_strategy.md`, `docs/topology_optimizer/ac/early_stopping.md`,
    `docs/topology_optimizer/ac/ac_loop.md`).

    - **Exact formulations as the yardstick.** DC optimal transmission switching is a MILP and NP-hard; you will
      formulate and solve it for a small grid and see where exact methods stop scaling, which is the reason
      ToOp's DC stage is a GPU heuristic.
    - **Acceptance.** `evaluate_acceptance` in `optimizer/ac/scoring_functions.py` compares each candidate's AC
      metrics with the unsplit grid's using multiplicative thresholds (defaults in
      `optimizer/interfaces/messages/ac_params.py`). Knowing exactly what it does, and where it misleads, is the
      difference between trusting and misreading ToOp's output.
    - **The DC winner can fail in AC.** The DC ranking in `res.json` comes from a linear model;
      `perform_ac_analysis` (`optimizer/benchmark/benchmark_utils.py`) re-runs the best DC topologies as full AC
      N-1 analyses and calls `evaluate_acceptance` against `unsplit_ac_metrics.json`, so the topology the DC
      stage ranked first can be rejected. Lab B has you check whether that happens in your own run.

## Topics

- Economic dispatch and DC optimal power flow (DC-OPF) as an LP; security-constrained OPF (SCOPF); locational
  marginal prices as dual values (revisits [Chapter 23](ch23-optimization-modeling.md)).
- **Optimal transmission switching (OTS) as a MILP**: big-M formulation, limits on the number of switched
  lines, security constraints; complexity of DC switching problems.
- Substation reconfiguration and busbar splitting: node-breaker MILPs, candidate identification.
- Awareness level (one exercise):
  - AC-OPF as a non-convex problem; convex relaxations: second-order cone (SOCP), quadratic convex (QC) and
    semidefinite (SDP); what a relaxation's bound certifies.
  - AC OTS as a mixed-integer nonlinear program (MINLP) and its mixed-integer SOCP relaxations.
  - Compact MILPs with PTDF/LODF (shift-factor) formulations instead of voltage angles.
  - Lazy (on-demand) generation of N-1 constraints: solve with few contingencies, add only the violated ones.
  - Benders and Lagrangian decomposition for SCOPF and switching problems.
- Heuristics and learning: greedy and sensitivity-based ranking, evolutionary search (ToOp), reinforcement
  learning ([Chapter 29](ch29-learning-based-control.md)).
- **Multi-fidelity search**: surrogate screening (DC) and high-fidelity validation (AC); candidate selection
  strategies (median, dominator, discriminator filters; [Chapter 27](ch27-pareto-multi-objective.md)); worst-k
  contingency early stopping; acceptance thresholds.
- Where topology optimization fits in European remedial action optimization processes: see
  [Chapter 21](../part-3-power-systems/ch21-coordinated-security-rao.md).

## Learning outcomes

- Formulate DC-OPF as an LP and DC optimal transmission switching as a MILP, solve both for a small grid, and
  report how the solve time grows with the number of switchable lines.
- Explain in one paragraph each what a convex relaxation, a shift-factor MILP, lazy N-1 constraint generation and
  Benders decomposition contribute to exact topology optimization.
- Reconstruct ToOp's AC accept/reject decision for a candidate from its metric files, and identify the cases in
  which the rule misleads (zero baselines, convergence vs overload, switching-distance mismatch).

## Resources

- Surveys and books:
  - Han, Xu & Hill, "Power Grid Topology Control", arXiv:2606.06995, 2026 (monograph-length survey).
  - van der Sar, Zocca & Bhulai, "Optimizing power grid topologies with reinforcement learning: a survey of
    methods and challenges", *Foundations and Trends in Electric Energy Systems* 9(1), 2025; arXiv:2504.08210.
  - Hedman, Oren & O'Neill, "A review of transmission switching and network topology optimization", IEEE PES
    General Meeting 2011.
  - Molzahn & Hiskens, "A Survey of Relaxations and Approximations of the Power Flow Equations", *Foundations and
    Trends in Electric Energy Systems* 4(1–2):1–221, 2019, doi:10.1561/3100000012, for the SDP, SOCP and QC
    relaxations.
  - Conejo, Castillo, Mínguez & García-Bertrand, *Decomposition Techniques in Mathematical Programming*
    (Springer, 2006), doi:10.1007/3-540-27686-6: Benders and Lagrangian decomposition (1 syllabus).
    **NOT ON IA**.
- Optimal power flow:
  - Frank & Rebennack, "An introduction to optimal power flow: Theory, formulation, and examples", *IIE
    Transactions* 48(12):1172–1197, 2016, doi:10.1080/0740817X.2016.1189626.
  - Chatzivasileiadis, "Lecture Notes on Optimal Power Flow (OPF)", arXiv:1811.00943.
  - Coffrin, Hijazi & Van Hentenryck, "The QC Relaxation: A Theoretical and Computational Study on Optimal Power
    Flow", *IEEE Trans. Power Systems* 31(4):3008–3018, 2016, doi:10.1109/TPWRS.2015.2463111.
- Transmission switching and busbar splitting:
  - Fisher, O'Neill & Ferris, "Optimal Transmission Switching", *IEEE Trans. Power Systems* 23(3):1346–1355,
    2008, doi:10.1109/TPWRS.2008.922256. **Core reading** for the MILP formulation.
  - Hedman et al. 2009, optimal transmission switching with contingency analysis
    ([Chapter 19](../part-3-power-systems/ch19-operational-security.md)).
  - Lehmann, Grastien & Van Hentenryck, "The Complexity of DC-Switching Problems", arXiv:1411.4369.
  - Kocuk, Dey & Sun, "New Formulation and Strong MISOCP Relaxations for AC Optimal Transmission Switching
    Problem", *IEEE Trans. Power Systems* 32(6):4161–4170, 2017, doi:10.1109/TPWRS.2017.2666718.
  - Ruiz, Foster, Rudkevich & Caramanis, "Tractable Transmission Topology Control Using Sensitivity Analysis",
    *IEEE Trans. Power Systems* 27(3):1550–1559, 2012, doi:10.1109/TPWRS.2012.2184777.
  - Goldis, Ruiz, Caramanis, Li, Philbrick & Rudkevich, "Shift Factor-Based SCOPF Topology Control MIP
    Formulations With Substation Configurations", *IEEE Trans. Power Systems* 32(2):1179–1190, 2017,
    doi:10.1109/TPWRS.2016.2574324.
  - Heidarifar & Ghasemi 2016 ([Chapter 13](../part-3-power-systems/ch13-substations-grid-models.md)); Heidarifar,
    Andrianesis, Ruiz, Caramanis & Paschalidis, "An optimal transmission line switching and bus splitting
    heuristic incorporating AC and N-1 contingency constraints", *IJEPES* 133:107278, 2021.
  - Morsy, Hinneck, Pozo & Bialek, "Security constrained OPF utilizing substation reconfiguration and busbar
    splitting", *Electric Power Systems Research* 212:108507, 2022.
  - Bastianel, Vanin, Van Hertem & Ergun, "Optimal transmission switching and busbar splitting in hybrid AC/DC
    grids", arXiv:2412.00270; Bastianel, Van Hertem, Ergun & Roald, "Identifying Best Candidates for Busbar
    Splitting", arXiv:2510.13000.
  - Rajaei & Cremer, "Security-Constrained Substation Reconfiguration Considering Busbar and Coupler
    Contingencies", arXiv:2603.04203.
- Learning-based topology control: see [Chapter 29](ch29-learning-based-control.md).
- ToOp papers: Westerbeck, van Dijk, Viebahn, Merz & Witthaut, "Accelerated DC loadflow solver for topology
  optimization", arXiv:2501.17529; Westerbeck, Hilfrich & Witthaut, "Transmission Topology Optimization using
  accelerated MapElites", arXiv:2605.10128.
- Documentation: PowerModels.jl <https://lanl-ansi.github.io/PowerModels.jl/stable/> (problem specifications
  include OTS; formulations include the SOC, QC and SDP relaxations); ToOp's `docs/topology_optimizer/metrics.md`,
  `docs/wall_of_shame.md` and the AC validation helpers in `optimizer/benchmark/benchmark_utils.py`.

## The AC stage in practice

Every statement below is checked against ToOp's source; the lab asks you to observe them in real output.

- **Worst-k early stopping.** With `early_stop_validation=True` (default), a pulled DC candidate is first
  evaluated in AC only on its contingency subset: the union of its DC `worst_k_contingency_cases` and those of the
  unsplit AC baseline (`optimizer/ac/evolution_functions.py`), plus the base case
  (`get_early_stopping_contingency_ids` and `score_strategy_worst_k` in `optimizer/ac/scoring_functions.py`). The
  unsplit results are restricted to the same subset before `evaluate_acceptance(..., early_stopping=True)`. Only
  survivors are evaluated on the remaining contingencies.
- **Acceptance rule.** A candidate is accepted when all hold: non-converging loadflows (minus its disconnected
  branches) $\le 1.0\times$ unsplit; `overload_energy_n_1` $\le 0.95\times$ unsplit; `critical_branch_count_n_1`
  $\le 1.1\times$ unsplit; and, only with `enable_critical_voltage_rejection=True`, voltage-jump and
  angle-difference counts $\le 1.1\times$ unsplit. The function returns `None` for accepted and a
  `TopologyRejectionReason` otherwise.
- **The zero-baseline trap.** Thresholds are multiplicative. If the unsplit grid has zero N-1 overload energy
  (or zero critical branches, or zero non-converging cases) on the evaluated subset, any positive value in the
  candidate is rejected, however small; on a worst-k subset this can happen even when the full grid is congested.
- **Convergence before overload.** The convergence check runs first. AC overload energy is summed only over
  branch results with valid values (`compute_overload_energy` drops NaN rows,
  `contingency/ac_loadflow_service/compute_metrics.py`). `docs/wall_of_shame.md` records that a topology which
  makes a previously diverging contingency converge, and thereby reveals overloads the unsplit run never
  reported, is rejected although it may improve the grid.
- **Non-convergence statuses.** `compute_metrics_single_timestep` counts `FAILED`, `MAX_ITERATION_REACHED` and
  PowSyBl's `NO_CALCULATION` as non-converging; `docs/wall_of_shame.md` notes that `NO_CALCULATION` can also mean
  "branch already disconnected".
- **Reduced outage set.** Bridges whose outage islands the grid are removed from the N-1 masks
  (`exclude_bridges_from_outage_masks` in `dc_solver/preprocess/preprocess.py`) because the LODF cannot
  handle them, and the AC stage uses the same reduced contingency list (`docs/wall_of_shame.md`).
- **Switching distance differs between stages.** DC sums a per-action `reassignment_distance` computed in
  preprocessing (`get_switching_distance` in `dc_solver/jax/aggregate_results.py`); AC counts the branch and
  injection reassignment differences of the realized topology (`extract_switching_distance` in
  `optimizer/ac/scoring_functions.py`). `docs/topology_optimizer/metrics.md` calls the mismatch a bug.
- **Unused parameter.** `early_stopping_non_convergence_percentage_threshold` is declared in
  `optimizer/interfaces/messages/ac_params.py` but no module reads it.

## Worked example

With generators $g$ at costs $c_g$, demands $d_n$, lines $\ell = (i, j)$ with susceptance $b_\ell$ and rating
$\bar f_\ell$, DC-OPF is the LP

$$
\begin{aligned}
\min_{P, f, \theta} \quad & \sum_g c_g P_g \\
\text{s.t.} \quad & \sum_{g \in G(n)} P_g - d_n
  = \sum_{\ell \in \delta^+(n)} f_\ell - \sum_{\ell \in \delta^-(n)} f_\ell && \forall n, \\
& f_\ell = b_\ell (\theta_i - \theta_j), \quad -\bar f_\ell \le f_\ell \le \bar f_\ell && \forall \ell, \\
& \underline P_g \le P_g \le \overline P_g \quad \forall g, \qquad \theta_{\text{ref}} = 0.
\end{aligned}
$$

Optimal transmission switching (Fisher, O'Neill & Ferris 2008) adds a binary $z_\ell$ (1 = in service) per
switchable line and replaces the line constraints by

$$
\begin{aligned}
& -\bar f_\ell z_\ell \le f_\ell \le \bar f_\ell z_\ell, \\
& -M_\ell (1 - z_\ell) \le b_\ell (\theta_i - \theta_j) - f_\ell \le M_\ell (1 - z_\ell), \\
& \textstyle\sum_\ell (1 - z_\ell) \le k .
\end{aligned}
$$

When $z_\ell = 1$ the flow equals $b_\ell(\theta_i - \theta_j)$; when $z_\ell = 0$ the flow is zero and the angle
difference is free up to $M_\ell / b_\ell$. As in [Chapter 23](ch23-optimization-modeling.md), a loose $M_\ell$
weakens the relaxation; a valid choice bounds $|b_\ell(\theta_i - \theta_j)|$ over all topologies, which is itself
non-trivial.

## Lab

!!! example "Lab A: DC-OPF and optimal transmission switching as MILPs"

    **Tools:** Jupyter with PuLP or Pyomo + HiGHS, NumPy, pandas and matplotlib; IEEE 30 data as in
    [Chapter 24](ch24-evolutionary-algorithms.md).

    1. Solve DC-OPF for IEEE 30 from the worked example and check the flows against your DC power flow. Print the
       duals of the nodal balance constraints (locational marginal prices) and explain where they differ.
    2. Add switching with the big-M formulation. Solve for $k = 1, 2, 3$ and for 5, 10, 20 and all 41 switchable
       lines; record cost, switched lines, solve time and MIP gap in a pandas table.
    3. Add N-1 security for the 5 most critical contingencies with your LODFs (as linear constraints in the flows),
       then add violated contingencies one at a time (lazy generation) until none remains. Record the iterations.
    4. Compare the switching decisions and the time with the brute-force and GA results of Chapter 24 and with the
       number of topologies per second ToOp reports in `res.json` (`total_branch_combis` and the run time).

!!! example "Lab B: the AC stage in practice"

    **Tools:** Jupyter with ToOp, pandas and pypowsybl ([Chapter 30](../part-5-computing/ch30-scientific-python.md));
    the run of `notebooks/example2_small_grid_toop.ipynb` from [Chapter 25](ch25-quality-diversity.md), or a fresh
    run of that notebook's `run_pipeline` call.

    1. **Reconstruct accept/reject.** For every `topology_N/` of the run, load `ac_metrics.json` and the run's
       `unsplit_ac_metrics.json` (both written by `save_ac_metrics_summary` in
       `optimizer/benchmark/benchmark_utils.py`) into one pandas table. Recompute each threshold check of
       `evaluate_acceptance` by hand, then call the function itself and compare. Note which topology the DC stage
       ranked first (`dc_info`) and whether it survived.
    2. **Probe the traps.** Build synthetic `Metrics` objects that show the zero-baseline trap and the
       convergence-before-overload rejection, and explain each result in one sentence.
    3. **Before/after diagrams.** Produce the before and after single-line diagrams of a split station as in the
       last cells of `notebooks/example2_small_grid_toop.ipynb` (mind voltage-level id vs station name:
       `save_slds_of_split_stations` passes the id to `get_single_line_diagram_custom` but names the file after
       the station) and highlight the switches that changed with its `highlight_grid_model_ids` argument.
    4. **Operator hand-off.** Open `orao_summary.json` (written by `save_orao_summary`, format in
       `changing_switches_to_orao_dict`, `optimizer/ac/summary.py`) and translate each entry into a switching
       instruction an operator could read. Compare the number of switch operations with `switching_distance` in
       `ac_metrics.json` and in `dc_info`.

## Exercises

1. Derive a valid big-M for a switchable line in a network where the line lies on a cycle of $m$ other lines with
   known ratings and susceptances. Why is a bound over all topologies harder? *(pen and paper)*
2. On a 3-bus network, show that switching off a line can lower the DC-OPF cost (Braess-like behavior), and read
   the explanation from the dual values. *(Jupyter: PuLP + HiGHS)*
3. Awareness: for each of SOCP/QC/SDP relaxation, shift-factor MILP, lazy N-1 constraints and Benders
   decomposition, write two sentences on what it would change if used to validate or seed ToOp's DC search, and
   which ToOp stage it would compete with. *(pen and paper)*
4. Write a two-page comparison of a MILP (with lazy N-1 constraints), a rule-based expert system and ToOp for the
   same operational question: which switching actions relieve a given N-1 overload in the next day's grid.
   Cover scalability, guarantees, AC feasibility and operator interpretability. *(pen and paper)*
5. Given `unsplit_ac_metrics.json` with `overload_energy_n_1 = 0.0`, `critical_branch_count_n_1 = 0` and
   `non_converging_loadflows = 2`, and a candidate with `0.4`, `1` and `1` (and no disconnected branches), which
   check rejects it first, and what would you change in the acceptance rule? *(pen and paper)*
