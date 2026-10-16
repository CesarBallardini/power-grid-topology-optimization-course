# Chapter 28 – Solutions

Solutions of the Lab and the Exercises of [Chapter 28](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`ieee30_ots.py`](src/ieee30_ots.py) (DC-OPF, optimal transmission
switching and lazy N-1 constraints with PuLP and HiGHS; it imports the IEEE 30 network data from Chapter 27's
`ch27_ieee30.py`), [`braess3.py`](src/braess3.py) (the 3-bus case of Exercise 2 and the big-M check of Exercise 1)
and [`acceptance.py`](src/acceptance.py) (a port of ToOp's AC acceptance rule and the table of a validation run).
Each module prints the numbers quoted below when run as a script. The port was checked against ToOp `ba1874e` with
[`test_acceptance_toop.py`](src/test_acceptance_toop.py) (ToOp's interpreter, `make test-toop`). Lab B uses two
fresh runs that [`lab_b_toop.py`](src/lab_b_toop.py) made in the course's ToOp container (image `toop-course:cpu`,
ToOp `ba1874e`, `docker run --cpus 4`): the `run_pipeline` call of `notebooks/example2_small_grid_toop.ipynb` with its
settings (30 s of DC search with ToOp's default seed 42, one descriptor `split_subs` with 4 cells, AC validation of
the 5 best topologies), once on the notebook's grid `data/grid_node_breaker/grid.xiidm` (44 s for the whole pipeline)
and once on the complex example grid `data/complex_grid/grid.xiidm` (65 s). Two settings differ from the notebook:
the preprocessing keywords that Pydantic drops are left out, and `switching_distance` is added to the observed
metrics so that `dc_info` carries it for step 4. The small result files of both runs are in
[`src/lab_b_run/`](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src/lab_b_run){ target="_blank" }
(`res.json`, `unsplit_ac_metrics.json`, and per topology `ac_metrics.json`, `orao_summary.json` and the
`switches.json` that `lab_b_toop.py` adds), and [`handoff.py`](src/handoff.py) reads them anywhere.

The IEEE 30 instance: pandapower's `case30` (MATPOWER data: 30 buses, 41 lines with ratings of 16 to 130 MW,
189.2 MW of load). MATPOWER's costs are quadratic, so the LP uses ten times the linear coefficient of each
generator: 20 $/MWh at bus 1 (80 MW), 17.5 at bus 2 (80 MW), 10 at bus 22 (50 MW), 32.5 at bus 27 (55 MW), 30 at
bus 23 (30 MW) and 30 at bus 13 (40 MW).

## Lab

### Lab A

1. **DC-OPF and LMPs.** The LP of the worked example costs 3101.0 $/h with the dispatch (57.5, 80, 50, 0, 1.7, 0)
   MW: the 10 $/MWh unit at bus 22 and the 17.5 unit at bus 2 run at their limits, bus 1 (20 $/MWh) is marginal,
   and 1.7 MW come from the 30 $/MWh unit at bus 23. The LP flows agree with `PTDF @ (generation - load)` to
   $10^{-13}$ MW. One line is at its rating, 22–24 (16 MW): the cheap unit at bus 22 can only leave through 21–22
   and 22–24, and 22–24 is full. The duals of the balance constraints range from 11.95 to 38.88 $/MWh. Without
   congestion every LMP equals the marginal unit's cost (with the ratings relaxed to 10 GW all thirty prices are
   20.00); with 22–24 binding, the buses behind it are cheap (bus 22: 11.95, bus 21: 12.60) and those in front of
   it expensive (bus 24: 38.88, buses 25–26: 32.54, bus 23: 30.00, the 30 $/MWh unit there being marginal for its
   corner of the grid). An LMP above the dearest running unit is possible: a MW delivered at bus 24 has to come
   from bus 23 or 27 *and* back off the cheap unit to keep 22–24 at 16 MW. The difference of the duals across a
   congested line is its shadow price.

    ```python title="ieee30_ots.py (data and DC-OPF)"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/ieee30_ots.py:data"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

2. **Switching with big-M.** The lines are ranked by their loading in the DC-OPF (22–24 at 100 %, 6–8 at 89 %,
   21–22 at 82 %, 2–6, 2–4, ...), and the first 5, 10, 20 or all 41 are switchable. With `mode='global'` the
   big-M of line $\ell$ is $b_\ell \sum_{k \ne \ell} \bar f_k / b_k$ (520 to 16,037 MW, valid for every
   connected topology); the cycle bound of [Exercise 1](#exercise-1) (`mode='cycle'`, 16 to 2,231 MW) is much
   tighter but valid only while the other lines stay closed. The sweep (HiGHS, 20 s limit):

    | switchable | $k$ | cost ($/h) | switched lines | time (s) | gap |
    |---|---|---|---|---|---|
    | 5 | 1, 2, 3 | 3095.5 | 2–6 | 0.05–0.10 | 0 |
    | 10 | 1, 2, 3 | 3095.5 | 2–6 (plus a no-effect line for $k \ge 2$) | 0.06–0.20 | 0 |
    | 20 | 1 | 3084.0 | 23–24 | 0.07 | 0 |
    | 20 | 2, 3 | 3084.0 | 4–6 + 23–24; 2–4 + 2–6 + 4–6 | 0.15–0.22 | 0 |
    | 41 | 1 | 3084.0 | 6–10 | 0.11 | 0 |
    | 41 | 2, 3 | 3084.0 | 6–7 + 6–10; 1–2 + 2–6 + 4–6 | 0.09–0.11 | 0 |

    Switching saves at most 17 $/h (0.5 %): it reroutes flow around 22–24 so that bus 1 can replace the 1.7 MW of
    the 30 $/MWh unit, and the dispatch becomes (59.2, 80, 50, 0, 0, 0). The optimum is degenerate (several lines
    give the same cost, and for $k \ge 2$ the solver adds lines whose opening changes nothing), so the "switched
    lines" column should be read as one optimal set among many. Every MILP, 41 binaries included, solves to a
    zero gap in a fraction of a second: on a 30-bus grid HiGHS's branch and bound sees nothing of the exponential
    growth, which only shows in the brute-force counts of step 4: with 41 binaries and no N-1 rows the model
    has a few hundred rows, and the solver's LP bounds prune almost all of the tree. The scaling appears on
    grids with hundreds of lines and N-1 constraints, where the
    literature of the chapter reports hours or time-outs; with 20 switchable lines and $k = 3$ the enumeration
    already has 1,351 topologies, with 41 lines 11,522.

    ```python title="ieee30_ots.py (switching MILP and brute force)"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/ieee30_ots.py:ots"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

3. **Lazy N-1 constraints.** For a fixed topology the post-outage flow is linear in the base flows,
   $f_k^{(c)} = f_k + \mathrm{LODF}_{kc} f_c$, so a contingency adds $2(n_\ell - 1)$ linear rows to the LP. The
   DC-OPF dispatch has a worst post-contingency loading of 155 % (22–24 after the outage of 27–28); the five
   worst contingencies by that measure are 27–28, 21–22, 6–8, 6–28 and 10–20. With those five the SCOPF costs
   3357.6 $/h and leaves one violation, 101.8 % on a line after the outage of 22–24 itself (index 30); adding it
   gives 3369.2 $/h with the dispatch (48.9, 80, 43.0, 17.2, 0, 0) and no violation: two iterations, six
   contingencies, instead of forty-one. The same LP with every non-bridge contingency at once gives the same
   3369.2 $/h, so the lazy loop lost nothing. Security costs 268 $/h here (8.6 %), far more than switching
   saves, which is why the chapter treats the SCOPF as the yardstick. With switching the LODFs would depend on
   the topology, so N-1 constraints inside the MILP need the angle formulation per contingency (one copy of the
   network per outage) or a shift-factor formulation that recomputes the factors per candidate topology
   ([Exercise 3](#exercise-3)); the lazy loop still applies, with a MILP solve per iteration.

    ```python title="ieee30_ots.py (LODF constraints and the lazy loop)"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/ieee30_ots.py:lazy"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

4. **Comparison.** Brute force over every set of at most two open lines solves 862 LPs in 3.1 s and finds the same
   3084.0 $/h (one line, 4–6, is enough; another of the degenerate optima); the MILP with $k = 2$ and all 41 lines
   takes 0.1 s. For $|D| \le 3$ the enumeration needs 11,522 LPs (about 40 s at this rate) and the MILP still a
   fraction of a second: the exact method wins by three orders of magnitude at this size, and a GA with the same
   budget as in Chapter 24 cannot beat an exact solver that finishes in 0.1 s. The Chapter 24 brute-force and GA
   figures of your own run and ToOp's `total_branch_combis` divided by its run time in `res.json` were **not
   available here**; ToOp's DC solver evaluates batches of thousands of topologies per second on a GPU, but with
   a different objective (N-1 overload energy of a fixed dispatch, no cost) and without optimality proof, so the
   comparison is between an exact answer to a small LP-based question and a fast approximate answer to a larger
   N-1 question.

### Lab B

1. **Reconstruct accept/reject.** `acceptance.run_table(run)` loads
   every `topology_N/ac_metrics.json` and the run's `unsplit_ac_metrics.json` (both `Metrics.model_dump()`,
   the first with `dc_info` added, written by `save_ac_metrics_summary`), recomputes the three default checks by
   hand (`convergence_ok`: non-converging loadflows minus disconnected branches $\le 1.0 \times$ unsplit;
   `overload_ok`: $\le 0.95 \times$; `critical_ok`: $\le 1.1 \times$), calls the ported rule for the verdict
   and ranks the rows by `dc_info.metrics.fitness` (`dc_rank` 1 is the DC winner). Under ToOp's interpreter,
   `evaluate_acceptance(Metrics(**split), Metrics(**unsplit))` on the same files must return `None` exactly on
   the rows whose three checks pass; `test_acceptance_toop.py` verifies this on a grid of 729 synthetic pairs, and
   `lab_b_toop.toop_verdicts` (tested in `test_lab_b_toop_toop.py`) gives the same verdict as the port for every
   stored topology. The node-breaker run made 684 epochs and kept 1 topology, the complex one 199 epochs and 4:

    | Grid | Topology | Actions | DC fitness | DC rank | AC overload energy (MW) | Critical branches | Non-converging | Rejected by |
    |---|---|---|---|---|---|---|---|---|
    | node-breaker | unsplit | `[]` | −61.445 | – | 16.336 | 2 | 3 | – |
    | node-breaker | 0 | `[5]` | −42.345 | 1 | 25.377 | 1 | 3 | `overload-energy` |
    | complex | unsplit | `[]` | −1310.860 | – | 1434.587 | 31 | 1 | – |
    | complex | 0 | `[1, 54, 68]` | −1160.366 | 1 | 1666.565 | 29 | 1 | `overload-energy` |
    | complex | 1 | `[54, 68]` | −1169.118 | 2 | 1680.170 | 29 | 1 | `overload-energy` |
    | complex | 2 | `[69]` | −1184.119 | 3 | 1480.865 | 30 | 1 | `overload-energy` |
    | complex | 3 | `[]` | −1243.819 | 4 | 1434.587 | 31 | 1 | `overload-energy` |

    The convergence and critical-branch checks pass everywhere; the overload check rejects every candidate, so
    **the DC winner does not survive on either grid**. On the node-breaker grid the split that removes 19 MW of DC
    overload energy adds 9 MW in AC. On the complex grid every split raises the AC overload energy above the unsplit
    1434.6 MW, although DC says each lowers it, and the AC order is nearly the reverse of the DC one: the DC winner has the
    second-highest AC overload, and the best AC result among the splits is the DC's third. Topology 3 is the
    empty genotype: no split, no disconnection, only `pst_setpoints` (`[-24, -6, -9]`; the four elites carry three
    different settings) that move its DC fitness from −1310.9 to −1243.8. `perform_ac_analysis` hands the runner
    only `actions` and `disconnections`, and `apply_topology_and_save` takes no setpoints, so the AC stage validates
    every topology with the grid's own taps: the AC metrics of this row equal the unsplit ones exactly, the 67 MW of
    DC improvement from the taps are never checked, and the row fails the threshold of 0.95 times itself. A
    multiplicative threshold of 0.95 asks for a 5 % improvement, so leaving the grid as it is counts as a rejection.

    ```python title="acceptance.py (the run table)"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/acceptance.py:table"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

2. **Probe the traps.** `acceptance.py` runs them (ToOp's own function returns the same objects in the test):

    - Zero baseline: unsplit `overload_energy_n_1 = 0.0`, candidate `0.01` → rejected, `overload-energy`,
      `value_after = 0.01`, `value_before = 0.0`, threshold 0.95. With an unsplit value of 1.0 MW the same
      candidate is accepted. *A multiplicative threshold on a zero baseline rejects every positive value, so on a
      worst-k subset where the unsplit grid happens to be clean no candidate can pass.*
    - Convergence before overload: unsplit `non_converging_loadflows = 1`, `overload_energy_n_1 = 0.0`; candidate
      `0` non-converging, `5.0` MW → rejected, `overload-energy`, 5.0 against 0.0. *The candidate made the
      diverging contingency converge and thereby revealed 5 MW of overload the unsplit run never computed; the
      rule compares that against the unsplit "zero" and rejects an improvement (`docs/wall_of_shame.md`).*
    - The other order: unsplit `1` non-converging and `100` MW; candidate `2` non-converging and `1.0` MW →
      rejected, `convergence`, 2 against 1. *The convergence check runs first, so a 99 % overload reduction is
      never looked at when one more case diverges.*

    ```python title="acceptance.py (the rule)"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/acceptance.py:rule"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

3. **Before/after diagrams.** `lab_b_toop.before_after` takes the switches of a topology's `orao_summary.json`,
   looks up their voltage-level *id* in the grid (not the station name: `save_slds_of_split_stations` passes the id
   to `get_single_line_diagram_custom` and names the file after the station), runs an AC load flow on the original
   and on the modified network, and calls `get_single_line_diagram_custom(net, level,
   highlight_grid_model_ids=switches)` on both. Node-breaker topology 0 splits `VL2`: the coupler `VL2_BREAKER`
   goes from closed to open, `L12_DISCONNECTOR_4_0` from closed to open and `L12_DISCONNECTOR_4_2` from open to
   closed, so line `L12` moves to the second busbar; the two SVGs have 45,735 and 46,475 characters and 3
   highlighted elements each. Complex topology 0 splits three voltage levels, `VL_2W_MV_HV_MV`, `VL_3W_HV` and
   `VL_MV`, with 3, 5 and 4 highlighted switches (12 in all); topology 3 changes no switch and has no diagram. To
   count the highlighted elements, count the elements whose `class` contains `highlight`, which is what ToOp adds
   to every matching id: the colour `#0050fcff` appears 7 times in every diagram, highlighted or not, because it
   sits in the CSS, once per voltage class.

    ```python title="lab_b_toop.py (ToOp's interpreter)"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/lab_b_toop.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.


4. **Operator hand-off.** `orao_summary.json` (`changing_switches_to_orao_dict`) lists the switches to open and
   to close, station by station, with nothing but `SWITCH` entries. `handoff.instructions` adds the kind and voltage
   level of each switch (`switches.json`) and writes one line per operation; for the node-breaker topology:

    1. Open breaker `VL2_BREAKER` in `VL2` (can break load current).
    2. Open disconnector `L12_DISCONNECTOR_4_0` in `VL2` (operate only without current).
    3. Close disconnector `L12_DISCONNECTOR_4_2` in `VL2` (operate only without current).

    An operator reads more into it than the file says: the disconnectors of `L12` may only move while the line
    carries no current, so its own breaker has to open first and close last, and that breaker is not in the list.
    `handoff.counts` compares the counts:

    | Grid | Topology | Actions | Operations | Breakers | Disconnectors | AC `switching_distance` | DC `switching_distance` |
    |---|---|---|---|---|---|---|---|
    | node-breaker | 0 | `[5]` | 3 | 1 | 2 | 3 | 2 |
    | complex | 0 | `[1, 54, 68]` | 12 | 4 | 8 | 12 | 8 |
    | complex | 1 | `[54, 68]` | 7 | 3 | 4 | 7 | 4 |
    | complex | 2 | `[69]` | 5 | 1 | 4 | 5 | 4 |
    | complex | 3 | `[]` | 0 | 0 | 0 | (not reported) | 0 |

    The number of operations equals the AC `switching_distance` (counted on the realized topology by
    `extract_switching_distance`) for every split, and the DC value (`dc_info`, a sum of per-action
    `reassignment_distance` values computed in preprocessing) equals the number of disconnector operations: it
    leaves out the coupler breakers that the splits open, one per split station and two in `VL_MV`
    (`VL_MV_BREAKER` and `VL_MV_BREAKER#0`). `docs/topology_optimizer/metrics.md` calls the mismatch a bug; for the
    hand-off, quote the AC number.

    ```python title="handoff.py"
    --8<-- "docs/part-4-optimization/ch28-topology-optimization/src/handoff.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Let line $\ell = (i, j)$ with susceptance $b_\ell$ lie on a cycle whose other $m$ lines $k = 1, \dots, m$ have
ratings $\bar f_k$ and susceptances $b_k$. When $\ell$ is open the two voltage angles are still linked by the rest
of the cycle: going from $i$ to $j$ along it,

$$
\theta_i - \theta_j = \sum_{k=1}^{m} \pm(\theta_{k}^{+} - \theta_{k}^{-}) = \sum_{k=1}^{m} \pm \frac{f_k}{b_k},
\qquad
|\theta_i - \theta_j| \le \sum_{k=1}^{m} \frac{\bar f_k}{b_k},
$$

because every line of the path respects its rating in any feasible point. The constraint
$-M_\ell(1 - z_\ell) \le b_\ell(\theta_i - \theta_j) - f_\ell \le M_\ell(1 - z_\ell)$ must not cut off any
feasible angle pair when $z_\ell = 0$ (then $f_\ell = 0$), so

$$
M_\ell = b_\ell \sum_{k=1}^{m} \frac{\bar f_k}{b_k}
$$

is valid, and it is the smallest valid value when the rest of the cycle is the only path between $i$ and $j$. In
the 3-bus ring of [Exercise 2](#exercise-2) (equal reactances 0.1 p.u., ratings 200, 200 and 100 MW) the bound
for line 1–3 is $M = b (200/b + 200/b) = 400$ MW; at the DC-OPF optimum with that line open the angle gap
$b(\theta_1 - \theta_3)$ is 360 MW, inside the bound (`braess3.big_m_cycle`, `gap_when_open`). A bound over all
topologies is harder because the other lines may be open too: the cycle may not exist, and the angle gap is then
set by whichever path survives, so the valid $M_\ell$ is the maximum over all connected topologies of the shortest
weighted path between $i$ and $j$ with weights $\bar f_k / b_k$, a combinatorial quantity. The safe choice is the
sum over *every* other line, $b_\ell \sum_{k \ne \ell} \bar f_k / b_k$ (a simple path uses each line at most
once), which `ieee30_ots.big_m(case, 'global')` uses and which is 10 to 100 times looser than the cycle bound
(520 to 16,037 MW against 16 to 2,231 MW on IEEE 30): a loose $M$ weakens the LP relaxation and slows the MILP,
as in Chapter 23.

```python title="braess3.py (big-M check)"
--8<-- "docs/part-4-optimization/ch28-topology-optimization/src/braess3.py:bigm"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

Three buses, three lines of equal reactance (1–2, 2–3 and 1–3, ratings 200, 200 and 100 MW), a 10 $/MWh
generator at bus 1, a 50 $/MWh generator at bus 3 and 180 MW of load at bus 3. With every line closed, two thirds
of a transfer from bus 1 to bus 3 take the direct line 1–3 and one third the path 1–2–3, so line 1–3 reaches its
100 MW rating when bus 1 produces 150 MW; the DC-OPF dispatches (150, 30) MW at 3000 $/h, flows (50, 50, 100) MW.
With line 1–3 open, all of bus 1's output takes 1–2–3, each line can carry 200 MW, and the DC-OPF dispatches
(180, 0) MW at 1800 $/h: opening a line lowers the cost by 40 %. The dual values explain it. Closed: LMPs
(10, 30, 50) $/MWh; the price at bus 3 is the cost of the expensive unit, which is marginal there, and bus 2 sits at
30 because a MW injected at bus 2 reaches bus 3 one third through 2–3 directly and two thirds via 2–1–3, loading
the congested line, so its value is a mix of the two generator prices. Open: LMPs (10, 10, 10), one price, the
cheap unit marginal everywhere, no congestion rent. The 1200 $/h saved are the 30 MW that the congestion forced
onto the expensive unit times the 40 $/MWh price difference. As in Braess's road-network paradox, removing a link
helps because flows follow impedances (Kirchhoff), not the cheapest path: the direct line attracts two thirds of
the transfer whether or not that is useful, and the DC-OPF has exactly that structure.

```python title="braess3.py (DC-OPF)"
--8<-- "docs/part-4-optimization/ch28-topology-optimization/src/braess3.py:opf"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch28-topology-optimization/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 3

- **SOCP / QC / SDP relaxation of the AC equations.** Used to *validate*, a relaxation gives a lower bound on the
  AC cost or a proof of infeasibility of a candidate topology in seconds, without a Newton solve that may diverge;
  it would complement the AC stage by rejecting infeasible candidates early rather than compete with the DC
  stage. Used to *seed*, the relaxation's angles and voltages are a better starting point than the DC flows, but
  a relaxation can be inexact (its solution need not correspond to a real operating point), so it cannot replace
  the AC validation that ToOp runs on the realized topology.
- **Shift-factor (PTDF/LODF) MILP.** It would replace ToOp's DC *search*: the same linear model ToOp evaluates
  by the million, written as a compact MILP without angle variables, solved to proven optimality for a small
  number of switching actions. It competes with the DC repertoire directly, with a guarantee instead of a
  population, and loses when the action set is busbar splits (the factors change per candidate topology and must
  be regenerated, which is what ToOp's BSDF/LODF updates do inside the solver).
- **Lazy N-1 constraint generation.** It would keep the exact method tractable: only the contingencies that
  actually bind are in the model (six of forty-one in Lab A3). In ToOp the analogue is the worst-k early stopping
  of the AC stage (`n_worst_contingencies`), which also looks at a subset first; a lazy exact method would compete
  with the DC stage for the proposal of candidates and with the early-stopping heuristic for the way contingencies
  are prioritized.
- **Benders decomposition.** The master picks the topology (binaries), the subproblem is the DC-OPF or the N-1
  check for that topology, and its duals return a cut to the master; the subproblem is exactly what ToOp's
  solver evaluates, so Benders would turn ToOp's evaluator into the oracle of an exact search and compete with
  the evolutionary DC stage. Its cost is one LP per candidate and a master MILP that grows with the cuts, so it
  scales to hundreds, not millions, of candidates.

## Exercise 4

**The question.** The day-ahead grid has an N-1 overload (line $\ell$ after outage $c$); the operator wants the
switching actions that remove it without creating others, by tomorrow morning.

**MILP with lazy N-1 constraints.** Formulation: DC-OPF or fixed-dispatch feasibility with binaries per switchable
line or busbar coupler, the big-M constraints of the worked example, and the post-contingency flow limits added
only for the contingencies found violated ([Lab A3](#lab-a)). *Scalability:* exact for tens of switchable
elements and a handful of binding contingencies (IEEE 30 solves in a fraction of a second); grows exponentially
in the worst case, and node-breaker busbar models multiply the binaries, so real grids need decomposition or a
restricted candidate list and still risk time-outs. *Guarantees:* optimality and infeasibility proofs for the DC
model and the contingencies in the set; nothing about contingencies not yet added, nothing about AC.
*AC feasibility:* none; the DC model ignores reactive power, voltages and losses, so every solution needs an
AC N-1 check afterwards, like ToOp's. *Interpretability:* high for the result (a list of switches with a proven
cost) and low for the reasoning (duals and cuts are not an operator's language), and a MILP's answer changes
abruptly with the data.

**Rule-based expert system.** Rules of the form "if line $\ell$ overloads after $c$, try the splits of the
substations adjacent to $\ell$, then the reconnection of parallel lines", encoded from operating experience and
checked by simulation before being proposed. *Scalability:* linear in the number of rules and simulations;
real-time capable; coverage limited to situations someone has written a rule for, and maintenance grows with
the grid. *Guarantees:* none of optimality; what it proposes has been simulated, so it is at least feasible in
the model used. *AC feasibility:* good when the rules are checked with an AC tool, which operators insist on.
*Interpretability:* the best of the three: every proposal comes with its rule, in the operator's vocabulary.

**ToOp.** A GPU DC solver evaluates millions of topologies, a MAP-Elites repertoire keeps the best per (number
of splits, switching distance), and an AC stage re-validates the best cells with an acceptance rule
([Lab B](#lab-b)). *Scalability:* the DC stage scales with GPU throughput, not with the number of binaries, and
handles busbar splits natively through BSDF/LODF updates; the AC stage is the bottleneck and is throttled by
worst-k early stopping. *Guarantees:* none; an evolutionary search can miss the optimum and its result depends on
seed and budget (Chapter 26), and the acceptance rule has the traps of Lab B2. *AC feasibility:* built in, every
proposal that reaches the operator passed a full AC N-1 with PowSyBl; the price is the mismatch between DC and AC
metrics (the DC winner can be rejected). *Interpretability:* medium: the output is a concrete list of switch
operations (`orao_summary.json`) with before/after metrics and diagrams, but *why* this split works is not
explained, and the weighted-sum fitness hides trade-offs (Chapter 27).

**Verdict.** For a next-day question with a few known overloads and a small candidate list, the MILP gives the
defensible answer and the expert system the fastest; ToOp is the tool when the candidate space is large (many
substations, splits rather than line switching) and AC feasibility must be part of the search rather than a
check afterwards. A reasonable process uses ToOp or the MILP to propose, the AC N-1 to validate, and keeps the
rule base as the operator's sanity check and the fallback when the optimizers time out.

## Exercise 5

The checks run in ToOp's order (`acceptance.py` reproduces it; ToOp's own function returns the same in the test):

1. Convergence: candidate $1 - 0 = 1$ non-converging loadflow $\le 1.0 \times 2$: passes.
2. Overload energy: $0.4 \le 0.95 \times 0.0 = 0$: **fails**. The function returns
   `criterion='overload-energy'`, `value_after=0.4`, `value_before=0.0`, `threshold=0.95`.
3. Critical branch count ($1 \le 1.1 \times 0 = 0$) would fail too, but is never reached.

This is the zero-baseline trap: the unsplit grid has no N-1 overload (perhaps only on the worst-k subset being
evaluated), so any candidate with a positive overload is rejected, however small, even though it also made one
diverging contingency converge, which is the improvement the operator would most want. Two changes make the rule
robust. First, compare with an absolute tolerance as well: accept when `split <= max(threshold * unsplit,
absolute)` with an absolute allowance in MW and in branches (for instance 1 MW and 1 branch), so that a zero
baseline does not become an impossible target. Second, give the convergence improvement credit: when the
candidate converges on cases the unsplit grid did not, compare the overload only on the cases both runs computed,
or recompute the unsplit baseline on the candidate's converging set; ToOp's `docs/wall_of_shame.md` records this
exact failure. The unused `early_stopping_non_convergence_percentage_threshold` parameter suggests such a rule
was planned.
