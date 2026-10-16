# Chapter 27 – Solutions

Solutions of the Lab and the Exercises of [Chapter 27](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`pareto.py`](src/pareto.py) (dominance, sorting, scalarizations,
indicators), [`dominator_filter.py`](src/dominator_filter.py) (a port of ToOp's dominator filter and its
counterexamples), [`repertoire_cells.py`](src/repertoire_cells.py) (clipped cells and the non-dominated set of a
repertoire table), [`augmecon.py`](src/augmecon.py) (AUGMECON2 with PuLP and HiGHS) and
[`ch27_ieee30.py`](src/ch27_ieee30.py) (the exact bi-objective front of IEEE 30 line switching, a small NSGA-II and a
small MAP-Elites). Each module prints the numbers quoted below when run as a script. The ports were checked against
ToOp `ba1874e` with [`test_dominator_filter_toop.py`](src/test_dominator_filter_toop.py) (ToOp's interpreter,
`make test-toop`). pymoo, moocore and QDax are not installed in the course environment, so every indicator and
sorting routine is implemented in NumPy and tested against brute force; Lab 5 and Exercises 3 and 4 also run them
in the labs container (`docker/labs`, `make test-labs`), with the `*_labs.py` modules. The ToOp results of Labs 1, 3,
4 and 7 come from fifteen DC runs that [`toop_front_toop.py`](src/toop_front_toop.py) made in the course's ToOp
container (image `toop-course:cpu`, ToOp `ba1874e`) on the complex example grid `data/complex_grid/grid.xiidm`, each
with 60 s of search and a CPU budget of `docker run --cpus 4` (three containers ran side by side, so a run took 61 to
70 s of wall time and used about 1.2 cores). Their `res.json` files are in
[`src/toop_front/`](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src/toop_front){ target="_blank" }
(about 13 KB each) and [`toop_front.py`](src/toop_front.py) prints every number quoted from them.

## Lab

1. **Pareto vs weighted-sum winner.** The run `fine_s1` uses ToOp's default repertoire (`split_subs` ×
   `switching_distance`, 5 × 45 cells, at most 4 splits), the default fitness $-$`overload_energy_n_1` and seed 1;
   it made 203 epochs in 60 s. `toop_front.table('fine_s1')` turns its `best_topos` into a table with the actions,
   `fitness`, the four metrics and the cell, and `lab1` computes the non-dominated set of (`overload_energy_n_1`,
   `switching_distance`, `split_subs`, `critical_branch_count_n_1`), all minimized. The metrics are rounded to
   $10^{-6}$ first: topologies with the same flows differ by about $10^{-12}$ MW (1160.3657499965468 against
   1160.365749996548), and without rounding a dominated elite looks non-dominated because its overload is smaller
   in the thirteenth digit. Of the 29 elites, **6 are non-dominated and 23 dominated**: a cell elite is only the
   best *of its cell*, and most cells are beaten by a cell with fewer splits or fewer switching operations. The
   best-fitness elite, ToOp's `max_fitness` (actions `[1, 54, 68, 118]`: 1160.37 MW, switching distance 10, 4
   splits, 31 critical branches) is **dominated**: `[1, 54, 68]` removes the same overload energy with 3 splits and
   switching distance 8. This is [Exercise 2](#exercise-2) on real data: the fitness gives weight zero to the other
   three metrics, so its maximizer is only weakly Pareto optimal, and which of the tied topologies wins is decided
   by rounding noise. No elite of `fine_s1` lies outside the 5 × 45 grid (switching distances reach 20, splits 4).
   With 5 × 9 cells (`coarse_s1`, 17 elites, 5 non-dominated) one elite sits in a clipped edge cell, and it is the
   weighted-sum winner again, `[1, 54, 68, 118]` with switching distance 10 in the cell of distance 8: under
   "exactly equal" constraints it would have no cell, and it is dominated anyway. The synthetic table of
   `repertoire_cells.synthetic_repertoire()` (made up, with up to 7 splits) shows the other axis: its 8 elites with
   `split_subs` $\ge 5$ all land in the edge cell of 4 splits, so the cell label hides that the winner uses 7.

    ```python title="toop_front.py (table and Lab 1)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/toop_front.py:table"
    ```


    ```python title="repertoire_cells.py (table and analysis)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/repertoire_cells.py:table"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

2. **Test the dominator filter.** `dominator_mask` ports `get_dominator_mask` line by line; the ToOp test runs
   the original on the same frames and the masks agree. Fitness is maximized, the observed metrics
   (`switching_distance`, `disconnections`) are minimized, the target metric is `split_subs`.

    (a) A non-dominated row that the filter removes:

    | row | fitness | split_subs | switching_distance | disconnections | dominator | Pareto |
    |---|---|---|---|---|---|---|
    | 0 | −1.0 | 1 | 3 | 0 | kept | non-dominated |
    | 1 | −2.0 | 1 | 2 | 1 | **removed** | **non-dominated** |

    Row 1 has a lower fitness than the group best (row 0) and is worse in one observed metric (`disconnections`),
    which is all the filter checks (an OR across metrics); that it is *better* in `switching_distance` is never
    looked at, so a genuine trade-off disappears.

    (b) Dominated rows that the filter keeps:

    | row | fitness | split_subs | switching_distance | disconnections | dominator | Pareto |
    |---|---|---|---|---|---|---|
    | 0 | −1.0 | 1 | 5 | 0 | kept | non-dominated |
    | 1 | −3.0 | 1 | 5 | 0 | **kept** | **dominated by row 0** |
    | 2 | −2.0 | 1 | 1 | 0 | kept | non-dominated |
    | 3 | −2.5 | 1 | 2 | 0 | **kept** | **dominated by row 2** |

    Row 1 has exactly the metrics of the best and a worse fitness: the filter asks for a *strictly larger* metric,
    so equality protects it. Row 3 is better than the best (row 0) in `switching_distance`, so the filter never
    removes it, although row 2, which is not the group best, beats it in fitness and in every metric. A Pareto
    filter compares every pair; the dominator filter compares each row with one row only.

    Order dependence: with `target_metrics = [split_subs, switching_distance]` and `disconnections` observed, the
    three rows of `order_example()` (fitness −1, −2, −3; `split_subs` 1, 1, 2; `switching_distance` 2, 3, 3;
    `disconnections` 0, 1, 2) give the mask `[True, False, True]`; with `[switching_distance, split_subs]` the
    mask is `[True, False, False]`. In the second order the pass over `switching_distance = 3` finds row 1 still
    present, makes it the group best and removes row 2 (lower fitness, more disconnections); in the first order
    row 1 has already been removed by the `split_subs` pass, row 2 is alone in its group and survives. Row 2 is
    dominated by row 0 in both cases, so neither answer is "right": the filter is a heuristic whose output depends
    on the order of the configuration list.

    ```python title="dominator_filter.py (the two filters)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/dominator_filter.py:filters"
    ```

    ```python title="dominator_filter.py (counterexamples)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/dominator_filter.py:examples"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

3. **Supported vs unsupported points.** Five more runs (`w1`, `w3`, `w10`, `w30`, `w100`, seed 1, 60 s each) use
   `target_metrics = [(overload_energy_n_1, 1), (critical_branch_count_n_1, w)]`; $w = 0$ is `fine_s1`.
   `toop_front.lab3` reports each run's winner in (overload energy MW, critical branches): $(1160.37, 31)$ for
   $w = 0$, $(1163.21, 27)$ for 1, $(1176.06, 27)$ for 3, $(1201.57, 24)$ for 10, $(1319.97, 18)$ for 30 and
   $(1359.25, 17)$ for 100. The six repertoires together have a front of 8 points, $(1160.37, 31)$, $(1163.21, 27)$,
   $(1190.23, 26)$, $(1201.57, 24)$, $(1266.73, 21)$, $(1316.57, 19)$, $(1319.97, 18)$ and $(1359.25, 17)$, of which
   **three are unsupported**: $(1190.23, 26)$, $(1266.73, 21)$ and $(1316.57, 19)$ (`supported_mask`; the test
   checks that no $w$ in $[0, 200]$ selects them). Each lies above the chord of its neighbours, so no weight can
   make it the fitness winner; the $\varepsilon$-constraint (minimize overload subject to at most 26, 21 or 19
   critical branches) returns each of them, and so does the achievement scalarizing function with a reference point
   1 MW and 1 branch below it. The runs also show the search's part: the $w = 3$ winner $(1176.06, 27)$ is dominated
   by $(1163.21, 27)$, which only the $w = 1$ run found, and for $w = 3$ and $w = 10$ the best point of the joint
   front under their own weight is $(1163.21, 27)$, not their winners. With 60 s per run a weight sweep mixes
   preference and search noise: compare the front of all runs, not the winners. In GeoGebra, plot the 8 points and
   the line `x + w*y = c` with sliders for `w` and `c`. The same mechanism on the five points of
   [Exercise 1](#exercise-1): 1001 weights $(w, 1 - w)$ select the points 0, 1, 3 and 4 and never $(4, 3.5)$.
   Geometrically, the iso-line $w f_1 + (1 - w) f_2 = c$ touches the front from below at a vertex of its convex
   hull; a point inside a dent is above the chord between its neighbours ($(4, 3.5)$ lies above the segment from
   $(3, 4)$ to $(6, 2)$, whose height at $f_1 = 4$ is $3.33$), so for every slope the chord's endpoints win first.
   In GeoGebra, draw the five points, the line `w*x + (1-w)*y = c` with sliders `w` in $[0, 1]$ and `c`, and lower
   `c` until the line meets the front: the contact point jumps from $(3, 4)$ to $(6, 2)$ as `w` passes
   $2/5$ and never stops at $(4, 3.5)$. The $\varepsilon$-constraint method does reach it:
   `epsilon_constraint(POINTS, 0, [0, 3.6])` (minimize $f_1$ subject to $f_2 \le 3.6$) returns $(4, 3.5)$, and so
   does the achievement scalarizing function with reference point $(4, 3)$; the Tchebycheff function with
   reference $(0, 0)$ and equal weights returns $(3, 4)$. The same happens to ToOp: `target_metrics` is a
   weighted sum, so a topology that cuts the overload a little less but has far fewer critical branches than its
   neighbours is never the fitness winner, however the weights are set.

    ```python title="pareto.py (scalarizations)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/pareto.py:scalarization"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

4. **Front quality.** `toop_front.lab4` compares the default resolution (`fine`, 5 × 45 cells) with a coarse one
   (`coarse`, 5 × 9) over seeds 1 to 5, on the bi-objective front (`overload_energy_n_1`, `switching_distance`) of
   each repertoire. A coarser `switching_distance` axis can only be made by lowering `num_cells`, which clips
   (`DescriptorDef.range` is declared in `optimizer/interfaces/messages/commons.py` but no module of `ba1874e` reads it). The
   fronts are normalized with the ideal and nadir points of the joint front of all ten runs, $(1160.37, 0)$ and
   $(1243.82, 8)$, and the reference point is $(1.1, 1.1)$. The joint front has four points, $(1160.37, 8)$,
   $(1169.12, 4)$, $(1184.12, 2)$ and $(1243.82, 0)$, and hypervolume 0.8364. Nine runs reach it exactly
   (hypervolume 0.8364, IGD+ and $\varepsilon$ zero); `fine_s4` misses its two lowest-overload points (it found
   $(1163.21, 8)$ and $(1172.16, 4)$ instead) and scores 0.8148, IGD+ 0.0176 and $\varepsilon$ 0.0364. Medians are
   equal and the exact two-sided Mann–Whitney test gives $p = 0.69$ for all three indicators, so the resolution
   makes no measurable difference on this grid with this budget: 60 s is enough for almost every run to find the
   same four trade-offs. The amount of work differs: the coarse repertoire is cheaper per epoch (17 elites against
   29) and makes 291 epochs (257–328) against 227 (203–246) in the same time ($p = 0.008$), which is why Chapter 26
   asks for budgets in evaluations. With the reference point at $(2, 2)$ the values change (3.6264, and 3.5741 for
   `fine_s4`) but not the ranking. Separating the two configurations would need a larger grid or a shorter budget.
   The indicators are in `pareto.py` (`hypervolume_2d`, `igd_plus`, `epsilon_indicator`, all tested against
   hand-computed values), and `ch27_ieee30.compare` also applies them to two optimizers on the IEEE 30 switching problem of step 6 over five seeds
   each, after normalizing with the ideal and nadir points of the exact front and with the reference point
   $(1.1, 1.1)$: the exact front has hypervolume 0.6486; the minimal NSGA-II reaches a median of 0.6418 (min
   0.6253, max 0.6486) and a median IGD+ of 0.0168 (0 to 0.0337); MAP-Elites with one cell per number of switched
   lines reaches 0.6486 and IGD+ $= 0$ on every seed. With five values per optimizer, report medians and ranges
   or an ECDF as in Chapter 26; a Mann–Whitney test on five against five can at best reach $p \approx 0.004$.
   Reference-point sensitivity ([Exercise 4](#exercise-4)): the hypervolume of the Exercise 1 front is 37.0 with
   the nadir as reference, 52.21 with $1.1 \times$ nadir and 262.0 with $2 \times$ nadir, so a comparison of two
   fronts is meaningful only with the same reference point, and the extreme points gain weight as the reference
   moves away.

    ```python title="toop_front.py (Labs 3 and 4)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/toop_front.py:weights"

    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/toop_front.py:quality"
    ```

    ```python title="toop_front_toop.py (the runs, ToOp's interpreter)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/toop_front_toop.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

5. **Swap in MOME.** Run in the labs container (`docker/labs`, QDax 0.5.0 on JAX 0.10.2, CPU only, limited to
   one core with `--cpus=1`) with [`mome_labs.py`](src/mome_labs.py) and the notebook
   [lab_labs.ipynb](src/lab_labs.ipynb). A ToOp run was not available, so the problem is the IEEE 30 line
   switching of step 6: at most three of the 41 lines switched off. The objectives are the two the lab names, the
   N-1 overload energy and the critical branch count (lines overloaded in at least one N-1 case), divided by their
   intact-network values (267.0 MW, 11 branches) and negated, because MOME maximizes. The descriptors stand in for
   ToOp's: the number of switched lines (`switching_distance`) and the number of buses at their ends
   (`split_subs`). Only 7 cells are reachable ($k$ lines touch between $k + 1$ and $2k$ buses), so the centroids
   are exactly those 7 points instead of a full grid.

    Every feasible switching (8,220 of the 11,522 sets; the others island a bus) is evaluated once on the host
    in 4.5 s, so the scoring function under `jit` is a table lookup and the exact Pareto front of every cell is
    known:

    | cell (lines, buses) | exact front: overload energy, critical branches |
    |---|---|
    | (0, 0) | 267.0 MW, 11 |
    | (1, 2) | 171.3 MW, 9 |
    | (2, 3) | 163.7 MW, 9 · 267.8 MW, 8 |
    | (2, 4) | 44.6 MW, 7 |
    | (3, 4) | 162.4 MW, 8 |
    | (3, 5) | 25.2 MW, 7 · 149.3 MW, 6 |
    | (3, 6) | 37.6 MW, 5 |

    Nine non-dominated points over seven cells: a scalar repertoire, one elite per cell, discards two of them,
    and both are trades of overload for one fewer critical branch, (267.8 MW, 8) next to (163.7 MW, 9) and
    (149.3 MW, 6) next to (25.2 MW, 7). That is the information the per-cell front adds. The exact MOQD score
    (sum of the per-cell hypervolumes, reference point $(1.1, 1.1)$ in units of the intact values) is 2.0353.

    Handling the three practical issues:

    - **Discrete genotype.** Three line indices sorted ascending and padded with the sentinel 41, as in
      Chapter 24. `MOMERepertoire` creates its empty slots with `jnp.zeros` in the default float dtype, so the
      genotype is stored as float32 and rounded back to integers in the scoring and mutation functions. The
      mutation is Chapter 24's in JAX (one slot set to the sentinel with probability 0.3 or to a random line,
      duplicates removed, re-sorted), and a child that islands the grid is replaced by its parent, so no stored
      fitness is infinite (QDax marks empty slots with $-\infty$).
    - **Memory of the per-cell fronts.** The repertoire holds cells × `pareto_front_max_length` × (genes +
      objectives + descriptors) numbers: here 7 × 8 × (3 + 2 + 2) × 4 bytes = 1,568 bytes. With ToOp's
      descriptor grid the cells are hundreds and the genotype holds the actions of every split substation, and
      the factor `pareto_front_max_length` multiplies all of it, which is why it is the first parameter to cut on
      a GPU. QDax 0.5.0 also keeps copies of the same point (equal points do not dominate each other) and, when a
      cell's front is longer than the maximum, truncates in index order rather than by hypervolume contribution.
    - **Hypervolume.** QDax's `compute_hypervolume` handles only two objectives and runs inside `jit`. For a
      third objective (here the number of switched lines, divided by 3) the global front is computed on the host
      with `moocore.hypervolume` after the run: 0.2546 for the exact front (5 points), reference 1.1 in each
      objective. QDax's helper also does not clip points beyond the reference point: it counts their area as
      negative, so `mome_labs.moqd_score` (which clips, like `pareto.hypervolume_2d`) and QDax's `moqd_score`
      metric agree on four seeds and differ on seed 1, where three stored points lie beyond the reference
      (1.8521 against 1.7200).

    MOME with a batch of 32 and 40 generations, that is 1,312 evaluations per run (11 % of the enumeration, the
    budget of step 6), on five seeds:

    | seed | MOQD score | exact points found | distinct points stored | 3-objective HV (moocore) | time |
    |---|---|---|---|---|---|
    | 0 | 1.8439 | 7 of 9 | 8 | 0.2546 | 6.0 s |
    | 1 | 1.8521 | 8 of 9 | 11 | 0.2546 | 3.6 s |
    | 2 | 1.7634 | 6 of 9 | 9 | 0.2525 | 3.5 s |
    | 3 | 1.7196 | 4 of 9 | 11 | 0.2442 | 3.4 s |
    | 4 | 1.9861 | 8 of 9 | 9 | 0.2546 | 3.7 s |

    Every seed covers all 7 cells; the median MOQD score is 1.8439, 90.6 % of the exact 2.0353, and the median
    run finds 7 of the 9 exact points, keeping 3 points that are not on the exact fronts (best found so far in
    a cell whose true front was missed). The first seed includes the `jit` compilation. The global
    three-objective front is easier than the per-cell fronts: three seeds reach the exact 0.2546. The per-cell
    trade-offs, the points the scalar repertoire would drop, are the hardest part to find in this budget.

    ```python title="mome_labs.py (exact table and cells)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/mome_labs.py:table"
    ```

    ```python title="mome_labs.py (MOME with QDax)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/mome_labs.py:mome"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

6. **Exact baseline.** IEEE 30 (pandapower's `case30`, 41 lines with MATPOWER ratings) with loads and generation
   scaled by 1.5 has an N-1 overload energy of 267.0 MW. Enumerating every set of at most three switched lines
   takes 11,522 evaluations and 3.7 s (0.32 ms each; the PTDF and LODF of the reduced network are rebuilt per
   set), so $|D| \le 4$ would add $\binom{41}{4} = 101{,}270$ sets, about 36 s. The exact bi-objective front
   (overload energy, number of switched lines) has one point per $k$:

    | switched lines | lines (1-based buses) | N-1 overload energy |
    |---|---|---|
    | 0 | – | 267.0 MW |
    | 1 | 25–27 | 171.3 MW |
    | 2 | 24–25, 6–28 | 44.6 MW |
    | 3 | 22–24, 24–25, 6–28 | 25.2 MW |

    The minimal NSGA-II (`nsga2`, population 24, mutation of Chapter 24, 1,200 evaluations, about 10 % of the
    enumeration) and MAP-Elites (`map_elites`, one cell per $k$) scored against this front are in step 4:
    MAP-Elites finds the exact front on all five seeds, NSGA-II misses the $k = 3$ point on some seeds (its
    population is spent on the dent between $k = 1$ and $k = 2$). The bi-objective facility location problem of
    Chapter 23 (opening cost against assignment cost) has the payoff table $(8, 20)$ and $(30, 9)$; AUGMECON2
    with a grid of 20 points on $f_2$ solves 7 MILPs (the bypass skips the grid points the slack already covers)
    and returns the six efficient points $(8, 20), (10, 18), (12, 15), (18, 13), (22, 11), (30, 9)$, exactly
    the non-dominated set of the seven opening decisions by enumeration. Two of them, $(10, 18)$ and $(18, 13)$, are
    unsupported (`supported_mask` on the front; a sweep of 1001 weights selects only the other four), which a
    weighted sum would miss.

    ```python title="ch27_ieee30.py (network and overload energy)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/ch27_ieee30.py:network"
    ```

    ```python title="ch27_ieee30.py (front, heuristics and comparison)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/ch27_ieee30.py:front"
    ```

    ```python title="augmecon.py"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/augmecon.py:augmecon"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

7. **Operator decision.** On the ToOp front of `fine_s1` in (overload energy, switching distance), the four points
   $(1160.37, 8)$, $(1169.12, 4)$, $(1184.12, 2)$ and $(1243.82, 0)$, `toop_front.lab7` finds the knee at
   $(1184.12 \text{ MW}, 2)$: one split (`[65]`) with two switching operations removes 60 MW of the 1243.82 MW of
   the empty genotype, and the 24 MW more that the best point removes cost six more operations. The pseudo-weights
   are $(1, 0)$, $(0.642, 0.358)$, $(0.488, 0.512)$ and $(0, 1)$ from the lowest overload to the lowest switching
   distance. The achievement scalarizing function on the three-objective front (overload, switching distance,
   splits), normalized by its ranges, with the reference "the best relief, at most 3 switching actions and 1 split",
   $(1160.37, 3, 1)$, picks $(1184.12, 2, 1)$, which is also the best topology that meets both limits. ToOp's a
   priori fitness picks $(1160.37, 10, 4)$: 2 % less overload energy for eight more switching operations and three
   more splits, and, as Lab 1 shows, not even a Pareto-optimal way to get it. The same method on the exact IEEE 30
   front of step 6, normalized with
   the ideal point $(25.2, 0)$ and the nadir $(267.0, 3)$: the knee (farthest point below the chord between the
   extremes) is $(44.6 \text{ MW}, 2 \text{ lines})$, where the second switched line removes 127 MW and the third
   only 19 MW more. The pseudo-weights are $(0, 1)$, $(0.373, 0.627)$, $(0.734, 0.266)$ and $(1, 0)$ for
   $k = 0, 1, 2, 3$: the $k = 2$ point is the one an operator who values overload about three times as much as
   switching effort would pick. The achievement scalarizing function with the reference point "no overload with
   one switched line", $(0, 1)$, returns the same $(44.6, 2)$: the reference is infeasible, and the point that
   misses it by the least in the worst objective is the knee. A posteriori selection shows the operator the whole
   front and the price of each step; ToOp's a priori weights pick one point before the search and, as step 3
   shows, cannot pick an unsupported one at all.

    ```python title="toop_front.py (Lab 7)"
    --8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/toop_front.py:decision"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

All five points are non-dominated: sorted by $f_1$, their $f_2$ values $9, 4, 3.5, 2, 1$ decrease strictly, so no
point is at least as good as another in both objectives. The supported points are the vertices of the lower-left
convex hull, $(1, 9), (3, 4), (6, 2), (9, 1)$; `supported_mask` confirms it by solving, for each point $p$, the
LP $\max t$ subject to $w \cdot (q - p) \ge t$ for every other point $q$, $w \ge 0$, $w_1 + w_2 = 1$, and $t \ge 0$
exactly for those four. The point $(4, 3.5)$ is unsupported: it lies above the segment from $(3, 4)$ to $(6, 2)$
(height $3.\overline{3}$ at $f_1 = 4$), so for every weight one of its neighbours has a smaller weighted sum. A
sweep over 1001 weights selects the indices $\{0, 1, 3, 4\}$ and never index 2. In GeoGebra: plot the points, the
line `w*x + (1 - w)*y = c` with sliders `w` $\in [0, 1]$ and `c`; as `w` decreases from 1 to 0 the contact point
moves $(1, 9) \to (3, 4) \to (6, 2) \to (9, 1)$ (at $w = 5/7, 2/5, 1/4$) without ever touching $(4, 3.5)$.

## Exercise 2

Let $x^*$ minimize $\sum_i w_i f_i(x)$ with every $w_i > 0$, and suppose some $y$ dominates it: $f_i(y) \le
f_i(x^*)$ for all $i$ and $f_j(y) < f_j(x^*)$ for some $j$. Then $\sum_i w_i f_i(y) < \sum_i w_i f_i(x^*)$, because
every term is at most the corresponding term of $x^*$ and the $j$-th is strictly smaller with $w_j > 0$, which
contradicts the optimality of $x^*$. So $x^*$ is Pareto optimal. With a zero weight the strict inequality can be
lost: for the points $(1, 2), (1, 3), (2, 1)$ and $w = (1, 0)$, both $(1, 2)$ and $(1, 3)$ minimize the weighted
sum ($= 1$), yet $(1, 3)$ is dominated by $(1, 2)$. It is only weakly Pareto optimal (no point is strictly better
in *every* objective). `weighted_sum_argmin` returns both indices for $w = (1, 0)$ and only $(1, 2)$ for
$w = (0.9, 0.1)$. This is why AUGMECON adds the slack term to the $\varepsilon$-constraint objective
([Lab 6](#lab)): a plain $\varepsilon$-constraint can also return weakly efficient points.

## Exercise 3

`fast_non_dominated_sort` builds the domination matrix $D_{ij}$ ($i$ dominates $j$) with broadcasting, counts the
dominators of every point and peels fronts: points with count zero form the current front, their dominated points
lose one count each, and so on (Deb et al. 2002). `crowding_distance` sorts each front per objective, gives the
boundary points infinity and adds the normalized gap between the two neighbours. In the course environment,
without pymoo, `test_pareto.py` compares the ranks with a brute-force peeling that applies the definition of
dominance pairwise, on 300 random points in three objectives (identical), and checks a case with duplicates. On
1,000 uniform random points in three objectives (seed 27) there are 20 fronts, 36 points in the first, 5 of them
with infinite crowding distance (two per objective, one point being extreme in two objectives).

The pymoo comparison runs in the labs container (`docker/labs`, pymoo 0.6.2) with
[`pymoo_sorting_labs.py`](src/pymoo_sorting_labs.py) and [exercises_labs.ipynb](src/exercises_labs.ipynb).
`NonDominatedSorting(method='fast_non_dominated_sort').do(points)` returns the fronts as lists of indices; turned
into ranks they are identical to `fast_non_dominated_sort` on the same 1,000 points (20 fronts, 36 in the first).
pymoo's `calc_crowding_distance` takes one front at a time and divides the sum of the normalized gaps by the number
of objectives, a constant factor that does not change any comparison inside NSGA-II; multiplied by 3, its distances
are identical to `crowding_distance` on every front of more than two points (the fronts of one or two points get
infinity in `pareto.py` by convention).

```python title="pareto.py (dominance and sorting)"
--8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/pareto.py:dominance"
```

```python title="pymoo_sorting_labs.py"
--8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/pymoo_sorting_labs.py:compare"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

Sort the non-dominated points by $f_1$; the dominated area is a staircase of rectangles, the $k$-th with width
$f_1^{(k+1)} - f_1^{(k)}$ (the reference point closing the last one) and height $r_2 - f_2^{(k)}$:

$$
HV = \sum_k \bigl(f_1^{(k+1)} - f_1^{(k)}\bigr)\bigl(r_2 - f_2^{(k)}\bigr).
$$

For the Exercise 1 points with the nadir $(9, 9)$ as reference, the two extreme points lie on the boundary and
add nothing, and $HV = 1 \cdot 5 + 2 \cdot 5.5 + 3 \cdot 7 = 37.0$; with $1.1 \times$ nadir $= (9.9, 9.9)$ it is 52.21 and with $2 \times$ nadir $= (18, 18)$
it is 262.0; a Monte Carlo estimate with 200,000 samples gives 262.95 for the last. The plot (not saved) of
$HV$ against the factor is a convex increasing curve: both the width of the last rectangle and the height of every
rectangle grow linearly with the factor, so $HV$ is quadratic in it, and the extreme points $(1, 9)$ and $(9, 1)$
contribute most of the growth. The property test in `test_pareto.py` checks monotonicity in the reference point
on random fronts.

The moocore comparison runs in the labs container (`docker/labs`, moocore 0.3.2) with
[`moocore_hypervolume_labs.py`](src/moocore_hypervolume_labs.py) and [exercises_labs.ipynb](src/exercises_labs.ipynb):
`moocore.hypervolume(points, ref=...)` gives 37.0, 52.21 and 262.0 for the three reference points, the same as the
rectangles. Over the factors 1.0, 1.1, ..., 2.0 the hypervolume is 37.0, 52.21, 69.04, 87.49, 107.56, 129.25,
152.56, 177.49, 204.04, 232.21 and 262.0, the quadratic curve to plot. On 200 random sets of 1 to 39 points the
largest difference between the two implementations is $8.9 	imes 10^{-16}$, rounding.

```python title="pareto.py (indicators)"
--8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/pareto.py:indicators"
```

```python title="moocore_hypervolume_labs.py"
--8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/moocore_hypervolume_labs.py:compare"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

The cell of a descriptor $(s, d)$ (`split_subs`, `switching_distance`) is `ravel_multi_index((s, d), (5, 45),
mode="clip")` $= 45 \min(s, 4) + \min(d, 44)$.

*Different cells, one dominates the other.* Topology A: 1 split, switching distance 2, overload 10 MW, cell
$45 + 2 = 47$. Topology B: 2 splits, distance 5, overload 12 MW, cell $90 + 5 = 95$. A dominates B in all three
quantities and both are stored, each as the elite of its cell. The repertoire is a table of "best for this
exact number of splits and switches", not a front: an operator who reads B's cell sees a valid elite and nothing
tells them that a cheaper topology does the same job better. The non-dominated set (`analyse`) has to be
extracted afterwards; on the synthetic table 19 of 30 elites are dominated.

*Same edge cell, different without clipping.* Topologies with descriptors $(4, 44)$ and $(4, 60)$ both land in cell
224; so does $(7, 44)$. Without clipping, $(4, 60)$ and $(7, 44)$ would have cells of their own (or none, under
exactly-equal constraints), and the cell "4 splits, 44 switches" would hold a topology with exactly those values.
With clipping, the elite of the last cell on an axis is the best topology with *at least* 4 splits or *at least* 44
switching operations, so an operator who takes the cell label at face value underestimates the effort of the
proposal, possibly by a lot. The label is a lower bound, and the only reliable numbers are the metrics stored with
the elite (`extra_scores`), which `table_from_res_json` reads.

```python title="repertoire_cells.py (cells)"
--8<-- "docs/part-4-optimization/ch27-pareto-multi-objective/src/repertoire_cells.py:cells"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch27-pareto-multi-objective/src){ target="_blank" } to see or download all the files of this chapter.
