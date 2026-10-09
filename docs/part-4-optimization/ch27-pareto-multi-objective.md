# Chapter 27. True Pareto multi-objective optimization

**Weeks:** 3 · **Semester:** 5 · **Prerequisites:** [Chapter 23](ch23-optimization-modeling.md), [Chapter 25](ch25-quality-diversity.md), [Chapter 26](ch26-experimental-methodology.md)

!!! info "Why it matters for ToOp"

    Operators trade off congestion relief against switching effort, number of split substations and other
    criteria. ToOp's `README.md` says its results are "stored as a pareto-front", and the optimizer paper
    (arXiv:2605.10128) speaks of "diverse candidate solutions on the pareto front". The code does something
    different, and students should be able to tell the two apart:

    - The DC fitness is a **weighted sum** (`optimizer/dc/genetic_functions/scoring_functions.py`), which can
      only reach *supported* Pareto points and hides trade-offs between the weighted metrics.
    - The MAP-Elites grid over (split substations, switching distance) acts like an **ε-constraint sweep** over
      those two criteria: interior cells fix the descriptor values exactly (equality constraints), while the last
      cell of each axis collects every larger value too, because `get_cell_index` uses
      `ravel_multi_index(mode="clip")` (`optimizer/dc/repertoire/discrete_me_repertoire.py`), so edge cells are
      "≥" constraints. Nothing ever extracts the non-dominated set, and many cell elites are dominated. `summarize`
      additionally drops every elite that does not beat the unsplit fitness.
    - The AC stage's **dominator filter** (`get_dominator_mask` in `optimizer/ac/select_strategy.py`) is a
      heuristic, not a non-dominated sort. For each grouping metric in `filter_dominator_metrics_target` and each
      discrete value of it, it takes the best-fitness row among the rows not yet removed, and removes every row of
      that group whose fitness is lower **and** which is worse in **any** observed metric (an OR across metrics).
      It processes the grouping metrics one after another on the shrinking set, so the result depends on their
      order. It can therefore **drop non-dominated rows** (worse fitness, better in one metric, worse in another)
      and **keep dominated rows** (for example, a row dominated by a row that is not its group's best, or with
      metrics equal to the best's). It runs after the optional discriminator and median filters.
    - There is no non-dominated sorting, crowding distance or hypervolume anywhere in the code.

## Topics

1. Pareto dominance (weak, strict); Pareto set and front; weakly, strictly and properly efficient points;
   ideal, utopian and nadir points; normalization.
2. **Scalarization**: weighted sum and its geometry (supported vs unsupported points; why non-convex parts of
   the front are unreachable); ε-constraint and AUGMECON/AUGMECON2; weighted Tchebycheff and achievement
   scalarizing functions; Normal Boundary Intersection.
3. Multi-objective integer and combinatorial optimization: unsupported non-dominated points; intractability.
4. **Multi-objective evolutionary algorithms**: Pareto ranking, NSGA-II (fast non-dominated sorting, crowding
   distance), SPEA2, MOEA/D (decomposition), SMS-EMOA and IBEA (indicator-based), NSGA-III (reference points).
5. **Performance indicators**: hypervolume (reference point, cost), GD, IGD, IGD+, additive ε-indicator;
   Pareto compliance. Comparing indicator values over repeated runs uses the budgets, ECDFs and tests of
   [Chapter 26](ch26-experimental-methodology.md).
6. Many-objective problems: loss of selection pressure, visualization.
7. **Multi-objective quality-diversity**: MOME (one Pareto front per cell; MOQD score = sum of per-cell
   hypervolumes), MOME-PGX.
8. Decision-making on the front: a priori, a posteriori and interactive methods; knee points; pseudo-weights;
   reference points; what suits TSO operators.
9. Power-system applications: multi-objective dispatch, transmission switching, congestion management,
   multi-objective reinforcement learning for topology control
   ([Chapter 29](ch29-learning-based-control.md)).

## Learning outcomes

- Prove or illustrate that weighted sums miss unsupported efficient points, and use ε-constraint or
  Tchebycheff scalarizations instead.
- Implement non-dominated sorting and crowding distance, and compute hypervolume and IGD+ for a front.
- Explain precisely how ToOp's repertoire cells and dominator filter differ from a Pareto filter, with
  counterexamples.
- Extract the true non-dominated set of ToOp's results, compare it with the weighted-sum winner, and argue
  which solutions to send to AC validation.

## Resources

- Required readings (all free):
  - Emmerich & Deutz, "A tutorial on multiobjective optimization: fundamentals and evolutionary methods",
    *Natural Computing* 17(3):585–609, 2018, open access, doi:10.1007/s11047-018-9685-y.
  - Emmerich & Deutz, *Multicriteria Optimization and Decision Making: Principles, Algorithms and Case
    Studies*, lecture reader of Leiden's MSc course, arXiv:2407.00359. **The best free textbook substitute.**
  - Deb, "Multi-objective optimisation using evolutionary algorithms: an introduction", KanGAL Report 2011003:
    <https://www.egr.msu.edu/~kdeb/papers/k2011003.pdf>.
  - Das & Dennis, "A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set
    generation in multicriteria optimization problems", *Structural Optimization* 14(1):63–69, 1997,
    doi:10.1007/BF01197559.
  - Deb, Pratap, Agarwal & Meyarivan, "A fast and elitist multiobjective genetic algorithm: NSGA-II", *IEEE
    TEC* 6(2):182–197, 2002, doi:10.1109/4235.996017.
  - Zitzler, Thiele, Laumanns, Fonseca & da Fonseca, "Performance assessment of multiobjective optimizers: an
    analysis and review", *IEEE TEC* 7(2):117–132, 2003, doi:10.1109/TEVC.2003.810758.
  - Pierrot, Richard, Beguir & Cully, "Multi-objective quality diversity optimization" (MOME), GECCO 2022,
    doi:10.1145/3512290.3528823; arXiv:2202.03057.
  - Lautenbacher, Rajaei, Barbieri, Viebahn & Cremer, "Multi-objective reinforcement learning for power grid
    topology control", IEEE PowerTech 2025, doi:10.1109/PowerTech59965.2025.11180236; arXiv:2502.00040.
- Optional readings:
  - Mavrotas, "Effective implementation of the ε-constraint method in Multi-Objective Mathematical Programming
    problems", *Appl. Math. Comput.* 213(2):455–465, 2009, doi:10.1016/j.amc.2009.03.037; Mavrotas & Florios,
    "An improved version of the augmented ε-constraint method (AUGMECON2) for finding the exact pareto set in
    multi-objective integer programming problems", *Appl. Math. Comput.* 219(18):9652–9669, 2013,
    doi:10.1016/j.amc.2013.03.002.
  - Zhang & Li, "MOEA/D", *IEEE TEC* 11(6), 2007; Beume, Naujoks & Emmerich, "SMS-EMOA", *EJOR* 181(3), 2007;
    Deb & Jain, "NSGA-III" Part I, *IEEE TEC* 18(4), 2014; Zitzler, Laumanns & Thiele, "SPEA2", TIK-Report
    103, ETH Zürich, 2001 (free PDF <https://sop.tik.ee.ethz.ch/publicationListFiles/zlt2001a.pdf>).
  - Guerreiro, Fonseca & Paquete, "The hypervolume indicator: computational problems and algorithms", *ACM
    Computing Surveys* 54(6), 2021; arXiv:2005.00515.
  - Miettinen, "Introduction to multiobjective optimization: noninteractive approaches" and Miettinen, Ruiz &
    Wierzbicki, "…interactive approaches", LNCS 5252, 2008.
  - Branke, Deb, Dierolf & Osswald, "Finding knees in multi-objective optimization", PPSN VIII, 2004.
  - Janmohamed, Pierrot & Cully, MOME-PGX, GECCO 2023; arXiv:2302.12668.
  - Ehrgott & Gandibleux, "A survey and annotated bibliography of multiobjective combinatorial optimization",
    *OR Spektrum* 22(4), 2000.
  - Abido, "Environmental/economic power dispatch using multiobjective evolutionary algorithms", *IEEE Trans.
    Power Systems* 18(4), 2003.
- Lectures and tutorials: Brockhoff, "Evolutionary Multiobjective Optimization" tutorial, GECCO 2019
  <http://www.cmap.polytechnique.fr/~dimo.brockhoff/publicationListFiles/broc2019a.pdf>; Brockhoff & Tušar,
  "Benchmarking Multiobjective Optimizers 2.0", GECCO 2023
  <http://www.cmap.polytechnique.fr/~dimo.brockhoff/publicationListFiles/bt2023a.pdf>; Brockhoff, EMO I and
  EMO II lectures, Université Paris-Saclay 2019
  <http://www.cmap.polytechnique.fr/~dimo.brockhoff/advancedOptSaclay/2019/slides.php>; MIT OCW ESD.77
  lecture 14 on multiobjective optimization.
- Software: pymoo (NSGA-II/III, MOEA/D, SMS-EMOA, `HV`, `IGD`, `IGDPlus`, `NonDominatedSorting`,
  `HighTradeoffPoints`, `PseudoWeights`, `ASF`) <https://pymoo.org>; moocore (hypervolume, ε-indicator,
  `is_nondominated`) <https://multi-objective.github.io/moocore/python/>; QDax MOME and NSGA-II/SPEA2 baselines
  (note: its hypervolume helper handles only two objectives); DEAP; DESDEO for interactive methods
  <https://desdeo.readthedocs.io>.
- Textbooks: the classic books are not usable on archive.org. Deb, *Multi-Objective Optimization using
  Evolutionary Algorithms* (2001; 2 syllabi); Coello Coello, Van Veldhuizen & Lamont, *Evolutionary
  Algorithms for Solving Multi-Objective Problems* (2002); Miettinen, *Nonlinear Multiobjective Optimization*
  (1999); Branke et al. (eds.), *Multiobjective Optimization: Interactive and Evolutionary Approaches* (2008):
  all **NOT ON IA** (withdrawn copies). Ehrgott, *Multicriteria Optimization*: **PRINT-DISABLED**. Borrowable:
  Coello Coello & Lamont (eds.), *Applications of Multi-Objective Evolutionary Algorithms* (2004) **BORROW**
  [`isbn_9789812561060`](https://archive.org/details/isbn_9789812561060).

## Reading guide

The graduate papers of this chapter are long; read them in this order and depth.

- **Mavrotas 2009 and Mavrotas & Florios 2013 (AUGMECON, AUGMECON2).** Required: the definition of the
  ε-constraint method, why plain ε-constraint can return weakly efficient points, the augmented objective with
  slack variables, the grid of ε values from the payoff table, and the bypass (early exit) rule that AUGMECON2
  adds. Skippable: payoff-table details beyond two objectives, solver-specific implementation listings and the
  large computational tables (read one table to see how exactness is reported).
- **Pierrot et al. 2022 (MOME).** Required: the problem statement of multi-objective quality-diversity, the
  repertoire with one Pareto front per cell, the addition rule, the MOQD score and the pseudocode. Skippable:
  the details of the benchmark environments and of the baselines' hyperparameters; skim the results for the
  metrics they report.
- **Guerreiro, Fonseca & Paquete 2021 (hypervolume survey).** Required: the definition of the hypervolume
  indicator, its Pareto compliance, the role and choice of the reference point, hypervolume contributions, and
  the two- and three-objective algorithms. Skippable: complexity proofs, many-objective exact algorithms and
  approximation schemes, unless you choose lab 5.
- **Zitzler et al. 2003.** Required: dominance-compliant comparison of fronts and why a single unary indicator
  cannot capture "better than" completely. Skippable: the formal proofs.

## Lab

!!! example "Lab menu: choose 2–3"

    **Tools:** Jupyter with NumPy, pandas, matplotlib and pymoo or moocore; ToOp results from
    [Chapter 25](ch25-quality-diversity.md); GeoGebra for the geometry in lab 3; PuLP + HiGHS for lab 6. These
    labs practice the chapter. A capstone project on multi-objective ToOp
    ([Chapter 35](../part-6-capstone/ch35-capstone.md)) must go beyond them, for example with a new selection
    strategy evaluated with the methodology of [Chapter 26](ch26-experimental-methodology.md).

    1. **Pareto vs weighted-sum winner.** Export ToOp DC results (`best_topos` in `res.json`) as a table of
       `overload_energy_n_1`, `switching_distance`, `split_subs`, `critical_branch_count_n_1` and fitness. Compute
       the non-dominated set (pymoo or moocore), mark the best-fitness topology, and count how many cell elites
       are dominated. Treat the clipped edge cells separately: which of their elites would lose their cell if the
       cells were "exactly equal" constraints?
    2. **Test the dominator filter.** Run `get_dominator_mask` from `optimizer/ac/select_strategy.py` next to a
       true Pareto filter on the same data. Build small pandas counterexamples of **both** kinds: (a) a
       non-dominated row that the filter removes (hint: worse fitness, better in one observed metric, worse in
       another); (b) a dominated row that the filter keeps (hint: observed metrics equal to the group best's, or
       a row dominated only by a row that is not its group's best). Then show that swapping the order of
       `target_metrics` changes the mask.
    3. **Supported vs unsupported points.** Sweep the weight `w` in
       `target_metrics = [(overload_energy_n_1, 1), (critical_branch_count_n_1, w)]`, collect the winners, and
       find non-dominated points no weight can reach. Draw the front and the weighted-sum iso-lines in GeoGebra
       with a slider for `w`. Repeat with an ε-constraint and an achievement scalarizing function.
    4. **Front quality.** Normalize, then compute hypervolume, IGD+ and the ε-indicator for two ToOp
       configurations (different descriptor resolutions); use at least 5 seeds per configuration and compare the
       indicator distributions as in Chapter 26. Discuss sensitivity to the reference point.
    5. **Swap in MOME.** Replace ToOp's scalar repertoire with QDax's `MOMERepertoire` (objectives: negated
       overload energy and critical branch count; descriptors unchanged). Handle the discrete genotype, GPU
       memory of per-cell fronts, and compute 3+-objective hypervolume with moocore outside JIT.
    6. **Exact baseline.** For the IEEE 30 line-switching problem of [Chapter 24](ch24-evolutionary-algorithms.md),
       enumerate all sets of at most 3 switched lines to obtain the exact bi-objective front (overload energy vs
       number of switched lines), and compare NSGA-II (pymoo) and MAP-Elites with it using hypervolume and IGD+.
       Separately, solve a bi-objective version of the facility location MILP of
       [Chapter 23](ch23-optimization-modeling.md) exactly with AUGMECON2 in PuLP.
    7. **Operator decision.** On a ToOp front, find knee points, pseudo-weights and an achievement-function
       solution for a reference point such as "at most 3 switching actions and 1 split"; discuss a posteriori
       selection vs ToOp's a priori weights.

## Exercises

1. For the points $(1, 9), (3, 4), (4, 3.5), (6, 2), (9, 1)$ (both minimized), find the non-dominated set, the
   supported points and the unsupported ones. Show with a GeoGebra slider on the weight that the weighted sum
   never selects the unsupported ones. *(GeoGebra)*
2. Prove that every optimum of a weighted sum with strictly positive weights is Pareto optimal, and give an
   example with a zero weight whose optimum is only weakly Pareto optimal. *(pen and paper)*
3. Implement fast non-dominated sorting and crowding distance in NumPy and test them against pymoo's
   `NonDominatedSorting` on 1,000 random points in 3 objectives. *(Jupyter: NumPy + pymoo)*
4. Compute the two-objective hypervolume of a front by sorting and summing rectangles, compare with moocore, and
   plot how the value changes when the reference point moves from the nadir point to 1.1 and 2 times the
   nadir. *(Jupyter: NumPy + matplotlib)*
5. With `n_cells_per_dim = (5, 45)`, give two topologies in different cells where one dominates the other in
   (`split_subs`, `switching_distance`, `overload_energy_n_1`), and two in the same edge cell that would be in
   different cells without clipping. What does each case mean for an operator reading the repertoire?
   *(pen and paper)*
