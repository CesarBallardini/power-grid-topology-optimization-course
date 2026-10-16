# Chapter 25 – Solutions

Solutions of the Lab and the Exercises of [Chapter 25](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`map_elites_cells.py`](src/map_elites_cells.py) (the cell index of
the worked example), [`small_grid_toop.py`](src/small_grid_toop.py) (runs ToOp's small-grid example with any
descriptors and weights; needs ToOp), [`repertoire.py`](src/repertoire.py) (reads `res.json`: elites, heatmap,
coverage, QD-score, and the cell lists of Exercises 2 and 5) and [`map_elites.py`](src/map_elites.py) (MAP-Elites
in NumPy on the Chapter 24 problem and the deceptive landscape of Exercise 3). The ToOp numbers were produced with
ToOp `ba1874e` on a laptop CPU (JAX on `TFRT_CPU_0`), 30 s per run; the output of the lab's step 2 is committed
as [`res_two_descriptors.json`](src/res_two_descriptors.json), trimmed to the fields the chapter reads, so that
`python repertoire.py` and the notebooks work without ToOp. `python map_elites.py` prints the numbers of Exercises
1 and 3.

## Lab

The small grid of `example2_small_grid_toop.ipynb` is tiny: `static_information_stats.json` reports 7 nodes, 9
branches, 11 N-1 cases (8 branch and 3 injection outages), 3 monitored branches, **one** relevant substation and
**10 actions**. The whole search space is ten topologies, and `split_subs` can only be 0 or 1; keep that in mind
when reading the repertoires below.

1. **Unchanged run.** `res.json` of the run (`small_grid_toop.py OUT step1`, the notebook's `ga_config`:
   `split_subs` with 4 cells, `runtime_seconds = 30`, seed 42): `initial_fitness = -61.445`
   (`initial_metrics.overload_energy_n_1 = 61.445` MW, `split_subs = 0`), `max_fitness = -42.345`,
   `iteration = 1142` epochs, `total_branch_combis = 913 608` evaluated topologies, and `best_topos` has one
   entry: action 5, one split, overload energy 42.345 MW. The first epoch alone evaluates 808 topologies, so the
   optimum is found at epoch 0 and the remaining 1 141 epochs change nothing. The run also writes 1 142 files
   `res_<epoch>.json` to `results/test/`.

    On Windows the DC stage crashes before the first epoch: `optimizer/dc/main.py` names the TensorBoard folder
    `str(datetime.now())`, whose colons are illegal in a Windows path (`WinError 123`). `small_grid_toop.py`
    patches the timestamp format on Windows; inside the container nothing is needed.

2. **Two descriptors and the plot.** With `me_descriptors = [split_subs: 4, switching_distance: 20]` and
   `plot = True`, `res.json` reports three elites, all with one split:

    | action | `split_subs` | `switching_distance` | overload energy [MW] | fitness |
    |---|---|---|---|---|
    | 3 | 1 | 0 | 57.058 | −57.058 |
    | 4 | 1 | 4 | 57.058 | −57.058 |
    | 5 | 1 | 2 | 42.345 | −42.345 |

    The run completed only 125 epochs (100 008 topologies) in the same 30 s, against 1 142 without the plot:
    `plot_repertoire` saves a PNG after every epoch (126 files in `results/test/plots/`), and drawing costs more
    than the optimization on this grid. In the heatmap the three cells sit on the row `split_subs = 1` in the
    columns 0, 2 and 4 of 20; every other cell is drawn at $1.05 \times (-57.058) = -59.9$, the "slightly below
    the worst finite fitness" value of `plot_repertoire`, so the 77 empty cells are nearly the same color as the
    two cells at $-57.058$ and only the cell at $-42.345$ stands out. Nothing marks empty cells as empty.
3. **Own heatmap, coverage, QD-score.** `repertoire.grid` fills a `(4, 20)` array with NaN and writes the fitness
   of each elite at `(split_subs, switching_distance)`, clipping indices like ToOp; `heatmap` masks the NaN cells
   so that they stay blank. Coverage $= 3 / 80 = 0.0375$. With the fitness shifted by the unsplit grid,
   QD-score $= (61.445 - 57.058) + (61.445 - 57.058) + (61.445 - 42.345) = 4.387 + 4.387 + 19.100 = 27.873$. The
   three cells are the only ones `summarize` reports; Exercise 4 discusses what the real archive held.
4. **Weights and cells.** Adding `critical_branch_count_n_1` to `target_metrics` changes every fitness by $-w
   \times$ (number of critical branches): the unsplit grid has 3 critical branches, actions 3 and 4 have 2 and
   action 5 has 1.

    | `target_metrics` | initial fitness | fitness of actions 3, 4, 5 | elites | epochs |
    |---|---|---|---|---|
    | overload energy only | −61.445 | −57.058, −57.058, −42.345 | 3, 4, 5 | 125 (with plots) |
    | + critical branches, $w = 1$ | −64.445 | −59.058, −59.058, −43.345 | 3, 4, 5 | 906 |
    | + critical branches, $w = 10$ | −91.445 | −77.058, −77.058, −52.345 | 3, 4, 5 | 926 |
    | overload energy only, cells `(2, 5)` | −61.445 | −57.058, −57.058, −42.345 | 3, 4, 5 | 895 |

    The set of proposed topologies does not change, and neither does their order: on a grid with ten topologies
    the archive is exhaustive after the first epoch, and the same action is best on both metrics, so a weight only
    rescales the fitness. The weights matter when two topologies trade one metric against the other (a split that
    removes one small overload while adding two critical branches elsewhere): a large $w$ then prefers the fewer
    critical branches, and a cell elite changes. The cell change `(2, 5)` keeps the same three elites because the
    descriptor values $\{0, 1\} \times \{0, 2, 4\}$ fit both grids; on the default `(5, 45)` repertoire of a real
    grid, fewer cells merge neighboring switching distances into one cell and the archive keeps only the best of
    each merged group, so the menu shrinks. To see any of this, rerun the lab on `data/case57_data_powsybl`
    ([Chapter 22](../../part-3-power-systems/ch22-import-preprocessing/index.md)), which has enough substations for several splits.

Solutions: `small_grid_toop.py` (steps 1, 2 and 4, under ToOp) and `main()` of `repertoire.py` (step 3); the
notebook [lab.ipynb](src/lab.ipynb) has the steps with empty cells.

```python title="small_grid_toop.py"
--8<-- "docs/part-4-optimization/ch25-quality-diversity/src/small_grid_toop.py"
```

```python title="repertoire.py"
--8<-- "docs/part-4-optimization/ch25-quality-diversity/src/repertoire.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch25-quality-diversity/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

`map_elites.py` reuses the fitness and the genotype of Chapter 24 (its `src` folder is put on the import path).
The descriptors are the number of switched lines (0 to 3, 4 cells) and the maximum N-0 loading after switching,
binned at 90, 100, 110 and 130 % (5 cells); the unswitched grid has 108.3 %, so it sits in cell $(0, 2)$. The
archive keeps the best genome per cell; each iteration mutates a random elite (crossed with another elite half the
time) and inserts the child if its cell is empty or worse. Same budget as the GA, 1 152 evaluations. Seed 0:

| evaluations | 32 | 100 | 300 | 600 | 1 152 |
|---|---|---|---|---|---|
| coverage | 0.20 | 0.50 | 0.65 | 0.65 | 0.70 |
| QD-score [MW] | −216.5 | −38.2 | 258.8 | 423.0 | 399.6 |
| best $E$ [MW] | 72.350 | 14.391 | 14.391 | 14.391 | 14.391 |

The QD-score starts negative because the first random cells are worse than the unswitched grid (a cell adds
$f_{\text{cell}} - f_{\emptyset}$, negative for a bad cell), and it can fall when a new cell is filled with a
bad topology (600 → 1 152), which is why ToOp's `summarize` drops cells that do not beat the unsplit grid. The
final archive, overload energy per cell (rows: lines switched, columns: loading bins):

| lines \ N-0 loading | < 90 % | 90–100 | 100–110 | 110–130 | ≥ 130 |
|---|---|---|---|---|---|
| 0 | – | – | 145.4 | – | – |
| 1 | – | – | 69.4 | 188.9 | 311.1 |
| 2 | 27.3 | 168.8 | 63.5 | 109.8 | 217.7 |
| 3 | 14.4 | 22.9 | 59.9 | 68.8 | 167.2 |

Over ten seeds the best cell is always $14.391$ MW (10 of 10 seeds, against 9 of 10 for the GA of Chapter 24 with
the same budget), coverage ends between 0.65 and 0.70 and the QD-score between 347.1 and 439.3 MW. Six cells are
never filled by any seed: the four cells $(0, \cdot)$ other than $(0, 2)$ cannot exist (no lines switched is one
grid, with one loading), and $(1, 0)$, $(1, 1)$ do not exist either: no single switch brings the maximum N-0
loading below 100 %. Coverage must be read against the reachable cells, 14 of 20 here. The archive is also a
readable answer in itself: the row "3 lines" shows the operator what the best topology is for each level of
remaining N-0 stress, the menu of solutions that ToOp's repertoire gives for `split_subs` and
`switching_distance`.

```python title="map_elites.py"
--8<-- "docs/part-4-optimization/ch25-quality-diversity/src/map_elites.py"
```

## Exercise 2

With `dims = (5, 45)` the flat index is $i \cdot 45 + j$ for `split_subs` $= i$ and `switching_distance` $= j$.
Exactly 2 split substations means row $i = 2$: cells $90$ to $134$, 45 cells (`repertoire.cells_with_value`).
Because `ravel_multi_index(..., mode='clip')` clips each descriptor to its last index, the last row holds every
topology with 4 **or more** splits (cells 180 to 224) and the last column every topology with 44 **or more**
switching operations (cells 44, 89, 134, 179, 224): $45 + 5 - 1 = 49$ cells whose descriptor values are lower
bounds, not exact values, cell 224 on both axes. The worked example's first topology, 1 split and 60 operations,
lands in cell 89 together with any one-split topology with 44 or more operations; the second, 7 splits and 3
operations, in cell 183 with the 4-, 5- and 6-split topologies with 3 operations.

## Exercise 3

`deceptive_fitness` on the integers $0 \le x < 100$: a wide hill $10\,e^{-((x-20)/15)^2}$ and a narrow peak
$20\,e^{-((x-85)/2)^2}$, with a valley between them where the fitness is below $0.01$ (at $x = 20, 50, 60, 70,
80, 83, 85$ it is $10$, $0.183$, $0.0082$, $0.0001$, $0.0386$, $7.36$, $20$). Both algorithms start with 20
points drawn from $0 \le x < 60$, mutate by a nonzero step of at most 10, and spend 2 000 evaluations.

- The elitist GA (tournament of two, two elites) ends at $x = 20$ in 20 of 20 seeds. Any child in the valley has
  a fitness below 0.2, loses every tournament against the hill and is never a parent; a child would have to jump
  from the hill ($x \le 30$) to $x \ge 83$ in one step to be selected, and the mutation cannot.
- MAP-Elites with the descriptor "which tenth of the domain $x$ is in" (10 cells) ends at $x = 85$ in 20 of 20
  seeds. A child at $x = 62$ has a terrible fitness but an empty cell, so it is kept; it becomes a parent as often
  as the hill's elite, its children fill the next cell, and after a few hops a child lands on the slope of the
  peak. The valley elites are **stepping stones**: they are not good solutions, they are the only route to one,
  and only an algorithm that rewards being different (occupying a new cell) rather than being better keeps them.

A detail worth knowing: with steps of at most 2 instead of 10, MAP-Elites also fails (0 of 20). The valley slopes
*down* towards the peak until $x \approx 80$, so the elite of each valley cell is the point nearest to the hill,
and children of that elite that move right are worse and never replace it; a cell is only crossed when one
mutation can jump from the elite to the next cell. Descriptors give the archive its memory, but the variation
operator still has to reach the next stepping stone.

```python title="map_elites.py (the deceptive landscape)"
--8<-- "docs/part-4-optimization/ch25-quality-diversity/src/map_elites.py:deceptive"
```

## Exercise 4

`summarize_repo` calls `filter_repo(repertoire, initial_fitness)` before converting cells to topologies, so
`best_topos` holds only the cell elites with a fitness strictly better than the unsplit grid. A coverage computed
from `res.json` is therefore a **lower bound** on the archive's coverage: in step 3 it counts 3 of 80 cells, but
every topology ToOp evaluated has a cell, including the unsplit grid itself (cell $(0, 0)$, dropped because it
does not beat itself) and the one-split actions that worsen the overload energy. On the small grid the filled
cells are at least $(0, 0)$ plus the cells of the nine split actions, whichever beat the unsplit grid or not. The
true coverage is not in `res.json`. Three ways to get it: (a) in the process that ran the optimizer, read
`repertoire.fitnesses` (the flat array of one fitness per cell, $-\infty$ for empty cells) and count the finite
entries; this is what `plot_repertoire` draws, so (b) the saved `repertoire_<epoch>.png` files show the true
filled cells, with the empty ones at $1.05 \times$ the worst finite fitness; or (c) re-evaluate every topology
of the search space with the DC solver, compute its descriptors and count the distinct cells, which is feasible
here (ten topologies) and is the brute force of Chapter 24. Whatever the route, report which definition you used:
"cells that beat the unsplit grid" and "cells ever filled" differ by exactly the elites that `filter_repo` drops,
and on a grid where few splits help the two numbers are far apart.

## Exercise 5

Three descriptors with 5, 45 and 3 cells give $5 \times 45 \times 3 = 675$ cells (`np.prod`), three times the
default 225. The fitness array is one `float64` per cell, $675 \times 8 = 5\,400$ bytes; with `cell_depth` $= d$
the repertoire keeps $d$ topologies per cell, $5\,400\,d$ bytes of fitness, and the genotypes (a few integers per
topology) dominate the memory anyway. The archive is not the problem; the picture is. `plot_repertoire`
(`optimizer/dc/repertoire/plotting.py`) reshapes the flat array to `n_cells_per_dim` and dispatches on its
length: a bar chart for one descriptor, a seaborn heatmap for two, and for anything else
`raise ValueError("Only 1D and 2D plots are supported")`, with a `TODO` for 3-D heatmaps. The test
`test_plot_repertoire_refuses_three_descriptors` in `test_small_grid_toop_toop.py` triggers that error with a
$(5, 45, 3)$ array. To look at a three-descriptor repertoire, slice it: one heatmap per value of
`disconnected_branches` (three $5 \times 45$ panels), which is also how an operator would read it, one menu per
number of disconnected branches.
