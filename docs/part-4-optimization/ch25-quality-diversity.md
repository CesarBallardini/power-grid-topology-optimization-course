# Chapter 25. Quality-diversity and MAP-Elites

**Weeks:** 2 · **Semester:** 4 · **Prerequisites:** [Chapter 24](ch24-evolutionary-algorithms.md), [Chapter 30](../part-5-computing/ch30-scientific-python.md)

!!! info "Why it matters for ToOp"

    The DC stage is a discrete MAP-Elites algorithm adapted from QDax
    (`optimizer/dc/repertoire/discrete_map_elites.py`, `optimizer/dc/repertoire/discrete_me_repertoire.py`). The
    whole loop runs on GPU in JAX.

    - **Descriptors.** Each cell of the repertoire is defined by integer descriptors. The defaults are
      `split_subs` with 5 cells and `switching_distance` with 45 cells (`me_descriptors` in
      `optimizer/interfaces/messages/dc_params.py`), so the default repertoire has $5 \times 45 = 225$ cells.
    - **Cell index.** Descriptors are mapped to a flat cell index with
      `jnp.ravel_multi_index(..., mode="clip")` (`get_cell_index` in `discrete_me_repertoire.py`). Values beyond
      the last cell are clipped into it, so the last cell of each axis means "at least this many", not "exactly".
    - **Fitness.** A weighted sum of metrics, `fitness = sum(-metric * weight)`
      (`optimizer/dc/genetic_functions/scoring_functions.py`); each cell keeps the best topology found for it
      (`cell_depth` > 1 keeps several).
    - **Output.** `summarize` in the same scoring module only reports cell elites whose fitness beats the unsplit
      grid, deduplicated. `plot_repertoire` (`optimizer/dc/repertoire/plotting.py`) draws a bar chart for one
      descriptor or a seaborn heatmap for two when `ga_config.plot` is set.

## Topics

- Why diversity: deceptive landscapes, novelty search, stepping stones.
- MAP-Elites: behavior descriptors, grid archive, per-cell elites, emitters; CVT-MAP-Elites.
- QD metrics: coverage, QD-score.
- Massively parallel QD on accelerators.
- ToOp's descriptors (`split_subs`, `switching_distance`, `disconnected_branches`) and what they mean for an
  operator: a menu of solutions of increasing switching effort.
- Discretization choices: number of cells per descriptor, clipped edge cells, and the growth of the archive as
  the product of the cell counts.

## Learning outcomes

- Implement MAP-Elites with a grid archive in NumPy, and compute coverage and QD-score over the run.
- Compute ToOp's cell index for a given descriptor vector, including clipped edge cells, and explain what a cell
  of the default repertoire represents.
- Run ToOp's small-grid example with two descriptors, plot the repertoire as a heatmap and explain how changing
  descriptors or `target_metrics` weights changes the set of proposed topologies.

## Resources

- Papers (core readings, all free):
  - Mouret & Clune, "Illuminating search spaces by mapping elites", arXiv:1504.04909, 2015.
  - Pugh, Soros & Stanley, "Quality Diversity: A New Frontier for Evolutionary Computation", *Frontiers in
    Robotics and AI* 3:40, 2016 (open access).
  - Cully & Demiris, "Quality and Diversity Optimization: A Unifying Modular Framework", *IEEE TEC* 22(2),
    2018; arXiv:1708.09251.
  - Vassiliades, Chatzilygeroudis & Mouret, CVT-MAP-Elites, *IEEE TEC* 22(4), 2018; arXiv:1610.05729.
  - Lim, Allard, Grillotti & Cully, "Accelerated Quality-Diversity through Massive Parallelism",
    arXiv:2202.01258.
  - Chalumeau et al., "QDax: A Library for Quality-Diversity and Population-based Algorithms with Hardware
    Acceleration", *JMLR* 25, 2024; arXiv:2308.03665.
  - **Westerbeck, Hilfrich & Witthaut, "Transmission Topology Optimization using accelerated MapElites",
    arXiv:2605.10128, 2026. The ToOp optimizer paper.**
- Online: QD community site <https://quality-diversity.github.io/>; QDax <https://github.com/adaptive-intelligent-robotics/QDax>
  and docs <https://qdax.readthedocs.io/>; pyribs tutorials <https://docs.pyribs.org/en/stable/tutorials.html>.
- No textbook covers quality-diversity, and no syllabus in the survey teaches it: this chapter is taught from
  papers.

## Worked example

With the default descriptors, `n_cells_per_dim = (5, 45)`. NumPy's `ravel_multi_index` behaves like the JAX
version ToOp uses:

```python
import numpy as np

split_subs = np.array([1, 7, 2])
switching_distance = np.array([60, 3, 44])
cells = np.ravel_multi_index((split_subs, switching_distance), dims=(5, 45), mode="clip")
print(cells)   # [ 89 183 134]
```

The first topology (1 split, 60 switching operations) lands in cell $1 \cdot 45 + 44 = 89$, the same cell as any
one-split topology with 44 or more operations. The second (7 splits, 3 operations) is clipped to row 4, cell
$4 \cdot 45 + 3 = 183$. Cells on the last row or column therefore compete over "at least" a value, which
matters when you read the repertoire as a trade-off curve ([Chapter 27](ch27-pareto-multi-objective.md)).

## Lab

!!! example "Lab: ToOp's repertoire as a heatmap"

    **Tools:** Jupyter with ToOp (container from [Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md)),
    NumPy, pandas and matplotlib.

    1. Run `notebooks/example2_small_grid_toop.ipynb` unchanged and open `res.json` in the run directory.
    2. The notebook overrides the default descriptors with a single `split_subs` descriptor of 4 cells and runs
       for 30 seconds. Change `ga_config` to use two descriptors, for example
       `[{"metric": "split_subs", "num_cells": 4}, {"metric": "switching_distance", "num_cells": 20}]`, set
       `"plot": True` and rerun. The heatmap is written by `plot_repertoire` to `plots/` inside the results
       `stats_dir`. Note that empty cells (fitness $-\infty$) are drawn with a value slightly below the worst
       finite fitness, so they look like bad cells.
    3. Build your own heatmap from `res.json` with pandas and matplotlib: one cell per
       (`split_subs`, `switching_distance`) pair from `best_topos[*].metrics.extra_scores`, colored by fitness,
       empty cells left blank. Compute coverage and QD-score (shift fitness by the unsplit `initial_fitness`).
    4. Change the `target_metrics` weights (for example add `("critical_branch_count_n_1", w)` for two values of
       `w`) and, separately, the number of cells. Explain how the set of proposed topologies changes.

!!! warning "The notebook's busbar-outage settings are silently ignored"

    The notebook builds `PreprocessParameters(action_set_clip=2**10, enable_bb_outage=True,
    bb_outage_as_nminus1=False)`. Neither `enable_bb_outage` nor `bb_outage_as_nminus1` is a field of
    `PreprocessParameters` (`interfaces/messages/preprocess/preprocess_commands.py`), and the model does not
    set `extra="forbid"`, so Pydantic drops both keywords without an error. Busbar outage data is
    preprocessed with `preprocess_bb_outages=True`; `enable_bb_outage` and `bb_outage_as_nminus1` belong to
    `ga_config` (`BatchedMEParameters` in `optimizer/interfaces/messages/dc_params.py`). The same keywords
    appear in `toop-engine-benchmark/benchmark_toop.py`. Check any configuration with
    `print(model.model_dump())`.

## Exercises

1. Implement MAP-Elites in NumPy for the line-switching problem of the [Chapter 24](ch24-evolutionary-algorithms.md)
   lab with descriptors "number of switched lines" (0–3) and "maximum N-0 loading after switching" binned into
   5 cells. Plot coverage and QD-score against evaluations and compare the best cell with your GA.
   *(Jupyter: NumPy + matplotlib)*
2. For `me_descriptors = (split_subs: 5 cells, switching_distance: 45 cells)`, list the cell indices that can
   hold a topology with exactly 2 split substations. Which cells can hold topologies that are not represented
   exactly by their descriptor values? *(pen and paper)*
3. Construct a one-dimensional deceptive fitness landscape where a GA with elitism converges to a local optimum
   but MAP-Elites with a suitable descriptor finds the global one. Explain the stepping-stone argument.
   *(Jupyter: NumPy)*
4. In the ToOp output, `summarize` drops every cell elite whose fitness does not beat the unsplit grid. What
   does this do to coverage computed from `res.json`, and how would you compute the true coverage?
   *(pen and paper)*
5. The default repertoire has 225 cells. Add `disconnected_branches` with 3 cells as a third descriptor: how many
   cells result, how many bytes do the fitness values alone need as float64, and why does
   `plot_repertoire` refuse to draw it? *(pen and paper)*
