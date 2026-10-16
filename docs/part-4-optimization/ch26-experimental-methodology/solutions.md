# Chapter 26 – Solutions

Solutions of the Lab and the Exercises of [Chapter 26](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`compare_runs.py`](src/compare_runs.py) (the rank-sum test,
$\hat A_{12}$ and ECDFs of the worked example), [`ga_experiment.py`](src/ga_experiment.py) (Lab steps 1 to 5 on
the GA of Chapter 24), [`toop_anytime.py`](src/toop_anytime.py) with [`toop_seeds.csv`](src/toop_seeds.csv) (Lab
step 6), [`rank_tests.py`](src/rank_tests.py) (Exercises 2 and 5), [`performance_profile.py`](src/performance_profile.py)
and its Octave twin [`perf_profile.m`](src/perf_profile.m) (Exercise 3). Every random number generator is seeded,
so every value below is reproducible except the wall-clock times, which are those of one laptop CPU; the ToOp runs
used ToOp `ba1874e` on CPU. `python ga_experiment.py`, `python rank_tests.py`, `python performance_profile.py` and
`python toop_anytime.py` print the numbers quoted here.

## Lab

1. **Protocol first.** Written before any run and printed with the results (`ga_experiment.protocol`):

    | element | choice |
    |---|---|
    | instance | IEEE 30 (MATPOWER `case30`), loads and generation × 1.4, switch off at most 3 lines ([Chapter 24](../ch24-evolutionary-algorithms/solutions.md#lab)) |
    | configurations | the Chapter 24 GA with crossover (probability 0.7) and without (0.0); everything else equal |
    | budget | 1 152 fitness evaluations per run, islanding candidates included |
    | measure | best N-1 overload energy at the end of the budget, in MW, lower is better |
    | repetitions | seeds 0 to 9 for each configuration |
    | test | two-sided Mann–Whitney U on the ten final values per configuration, $\alpha = 0.05$ |
    | effect size | Vargha–Delaney $\hat A_{12}$ and a bootstrap 95 % confidence interval of the difference in medians |

    Nothing in this list may change after the runs. In particular the measure is the value *at the end of the
    budget*; the evaluations needed to reach the optimum are not in the protocol, so they cannot become the
    headline when the final values disagree with the hypothesis.

2. **Run.** `run_all` returns one pandas row per (configuration, seed, evaluation) with the best-so-far value and
   the elapsed seconds, $2 \times 10 \times 1\,152 = 23\,040$ rows; `switching_ga.Evaluator` records that history
   on every call. Before the first finite evaluation of a run (every candidate islanded so far) the best value is
   $+\infty$; the curves start at the first finite value. The runs took 0.42 to 0.80 s each on this machine.
3. **Anytime curves.** `anytime` interpolates every run onto a common grid of 60 points and takes the 25, 50 and
   75 % quantiles over the ten seeds; `plot_anytime` draws the median line and the quantile band per
   configuration, against evaluations and, with `x='seconds'`, against time. Against evaluations the two bands
   overlap almost entirely: at 313 evaluations the medians are 28.96 MW (crossover) and 28.85 MW (no crossover),
   with quartiles 26.55–32.05 and 16.53–32.91. The wider band without crossover says that its seeds are more
   spread out, not that it is better. Against seconds the picture depends on the machine: the time per evaluation
   is the same for both configurations (the fitness dominates), so the curves against seconds are the curves
   against evaluations stretched by each run's speed, and the bands widen by the timing noise.
4. **ECDFs and tests.** Final values (MW), seeds 0 to 9:

    | configuration | final $E$ per seed | median |
    |---|---|---|
    | crossover | 14.391 ×9, 22.943 (seed 4) | 14.391 |
    | no crossover | 14.391 ×7, 22.943 (seeds 7 and 9), 22.943 | 14.391 |

    Both ECDFs are two-step functions: a jump at 14.391 (to 0.9 and 0.7) and the rest at 22.943. Mann–Whitney
    $U = 45$, $p = 0.58$; $\hat A_{12} = 0.55$ (a crossover run beats a run without crossover 55 % of the time,
    ties counted half); bootstrap 95 % interval of the difference in medians $(-4.28, 0.00)$ MW, which contains 0.
    Conclusion, in two sentences: *With 1 152 evaluations on this instance, the data do not show a difference
    between the GA with and without crossover (ten seeds each, $p = 0.58$, $\hat A_{12} = 0.55$, 9 against 7 runs
    at the optimum). A difference of one or two seeds reaching the optimum is within what chance produces, and
    nothing larger was observed.* Not "crossover helps" and not "crossover does not help": the experiment has no
    power to see a small effect (Exercise 5).

5. **Wall-clock trap.** The runs are too short (half a second) to share the machine with a real burner process in
   a controlled way, so `wall_clock_budget` simulates it: the recorded times of the even seeds are multiplied by 3,
   the budget is half the median run time (0.24 s here) and each run's value is its best-so-far at that moment.
   On a quiet machine the budget favours the configuration that happens to run faster on this machine: medians
   14.391 against 26.664 MW, $p = 0.12$, $\hat A_{12} = 0.69$, a difference that the evaluation budget did not
   show, produced by the clock. With half the runs slowed the medians are 26.775 against 27.343 MW, $p = 0.59$,
   $\hat A_{12} = 0.57$: the slowed runs completed a median of 194 evaluations in the budget against 559 for the
   others, so the result now measures which seeds were slowed, not which configuration is better. Every number in
   this step changes from one machine, or one run, to the next, which is the point: a wall-clock budget confounds
   the optimizer with the hardware and the load, and only an evaluation budget (or a per-evaluation cost model)
   makes two runs comparable. This is exactly ToOp's situation, whose budget is `runtime_seconds`.

6. **Optional, ToOp.** Five runs of the small-grid example with `random_seed` 1 to 5 and `runtime_seconds = 30`
   (`small_grid_toop.py` of Chapter 25, which writes a `res_<epoch>.json` per epoch), collected by
   `toop_anytime.collect` and committed, compacted, as `toop_seeds.csv`:

    | seed | epochs in 30 s | `total_branch_combis` | final `max_fitness` | first epoch at the maximum |
    |---|---|---|---|---|
    | 1 | 1 307 | 1 045 608 | −42.345 | 0 |
    | 2 | 909 | 727 208 | −42.345 | 0 |
    | 3 | 1 143 | 914 408 | −42.345 | 0 |
    | 4 | 1 022 | 817 608 | −42.345 | 0 |
    | 5 | 1 210 | 968 008 | −42.345 | 0 |

    The same 30 s bought between 909 and 1 307 epochs (a ratio of 1.44) on one idle laptop: the number of epochs,
    not the seconds, is the budget a run actually received, and `total_branch_combis` (808 topologies per epoch
    here) is the comparable count. The anytime curves against epochs or against evaluated topologies are flat
    lines of different lengths at $-42.345$: the small grid has ten topologies and the first epoch evaluates
    them all, so every seed is at the optimum from epoch 0. `truncate` cuts every run at the smallest common
    epoch (909 epochs, 720 808 topologies): the five runs are then compared under an equal budget, and here they
    are identical. On a real grid the curves climb for many epochs, and the truncation decides which run looks
    best.

Solutions: `main()` of `ga_experiment.py` (steps 1 to 5) and of `toop_anytime.py` (step 6); the notebook
[lab.ipynb](src/lab.ipynb) has the steps with empty cells.

```python title="ga_experiment.py"
--8<-- "docs/part-4-optimization/ch26-experimental-methodology/src/ga_experiment.py"
```

```python title="toop_anytime.py"
--8<-- "docs/part-4-optimization/ch26-experimental-methodology/src/toop_anytime.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-4-optimization/ch26-experimental-methodology/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Three things wrong:

1. **The best of $n$ is not a performance measure.** The maximum of 20 draws grows with 20 and with the variance,
   so a method with a wider spread wins this comparison even when its median is worse; the baseline may have
   found a nearly identical topology in 19 of 20 runs and lost on the extreme value. The best run is one sample
   of the random process, and the paper reports neither the distribution it came from nor how the other 19 runs
   ended.
2. **No test, no effect size, no uncertainty.** "1 of 20 against 0 of 20" is a difference of one event; the
   two-sided Fisher test on that table gives $p = 1$, and no confidence interval is given. Nothing in the report
   lets a reader tell luck from an effect.
3. **The conditions are not stated.** Which budget (evaluations or seconds, and on what hardware), which seeds,
   which instance(s), whether the 20 runs were all the runs made or the 20 kept, and whether "best topology" was
   defined before the runs. Without the configuration and the seeds the result cannot be reproduced; without the
   budget it cannot be compared; a wall-clock budget on two machines (Lab step 5) can produce the reported
   sentence on its own.

A better table reports the whole distribution and the design, for example

| method | runs | budget | median final $E$ [MW] | 25–75 % | best | worst | runs at target | $\hat A_{12}$ against baseline | $p$ (rank-sum) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 20 | 1 152 evaluations, seeds 0–19 | … | … | … | … | 0/20 | 0.5 | – |
| new method | 20 | same | … | … | … | … | 1/20 | … | … |

with the anytime median curves and the ECDFs of the final values as figures, the instance, the configuration
files, the seeds and the hardware listed, and the sentence about the best run, if kept at all, placed after the
median. Where the instances are many, one line per instance and the Friedman test of Exercise 2.

## Exercise 2

`rank_tests.synthetic_instances` builds the table: eight instances with a random difficulty between 20 and 60
MW, three configurations with noise $\sigma = 3$ MW, five seeds each and the median per instance, configuration C
shifted by $-1$ MW (better). The observations are paired by instance, so the Friedman test ranks the three
configurations within each instance and tests whether the mean ranks differ:

| | A | B | C |
|---|---|---|---|
| mean rank | 2.12 | 2.12 | 1.75 |

Friedman $\chi^2 = 0.75$, $p = 0.69$: no evidence of any difference. The post-hoc comparisons would normally stop
here; running them anyway shows the correction at work. Wilcoxon signed-rank $p$-values of the three pairs, Holm
step-down (sorted $p_i$ compared with $\alpha / (m - i + 1)$, adjusted $p$ the running maximum of $(m - i + 1)
p_i$):

| pair | $p$ | threshold | Holm-adjusted $p$ | reject |
|---|---|---|---|---|
| A vs C | 0.1484 | 0.0167 | 0.4453 | no |
| B vs C | 0.1953 | 0.0250 | 0.4453 | no |
| A vs B | 0.9453 | 0.0500 | 0.9453 | no |

A 1 MW advantage with 3 MW of noise, eight instances and five seeds per instance is invisible to this design
(with 0.3 MW it is $p = 0.88$). The rank tests are honest about that: with eight paired observations the smallest
possible two-sided signed-rank $p$-value is $2 / 2^8 = 0.0078$, so after Holm's $0.05 / 3$ a pair can only be
rejected when C wins on every instance. More instances, not more seeds per instance, is what raises the power
here, because the test counts instances.

```python title="rank_tests.py"
--8<-- "docs/part-4-optimization/ch26-experimental-methodology/src/rank_tests.py"
```

## Exercise 3

`performance_profile.synthetic_times` gives a seeded table of times to reach the target for 20 instances: the GA
is fast but fails on 7 instances (NaN), MAP-Elites is 1.5 to 2.5 times slower than the base time and never
fails, annealing is the fastest on the easy half and four times slower on the hard half. For solver $s$ and
instance $p$ the ratio is $r_{ps} = t_{ps} / \min_s t_{ps}$, $+\infty$ for a failure, and

$$
\rho_s(\tau) = \frac{1}{n_p}\,\bigl|\{p : r_{ps} \le \tau\}\bigr|,
$$

a failure never counting, not even for $\tau = \infty$ (`inf <= inf` is true in NumPy and in Octave, so the
code masks the infinite ratios explicitly).

| $\tau$ | GA | MAP-Elites | annealing |
|---|---|---|---|
| 1 | 0.30 | 0.20 | 0.50 |
| 2 | 0.65 | 0.35 | 0.60 |
| 3 | 0.65 | 0.60 | 0.70 |
| 5 | 0.65 | 1.00 | 0.95 |
| 10 | 0.65 | 1.00 | 1.00 |
| $\infty$ | 0.65 | 1.00 | 1.00 |

$\rho_s(1)$ is the fraction of instances on which $s$ was the fastest (ties count for every tied solver):
annealing wins half of the instances, the GA 30 %, MAP-Elites 20 %. For large $\tau$ every solver reaches the
fraction of instances it solved at all: 1.0 for MAP-Elites and annealing, 0.65 for the GA, whose seven failures
are visible as the gap to 1 that no $\tau$ closes. In between the curves cross: the GA is within a factor 2 of
the best on 65 % of the instances, where MAP-Elites is at 35 %, but MAP-Elites reaches 100 % at $\tau = 5$
(its largest finite ratio is 3.81) while annealing needs $\tau = 10$ (largest ratio 5.18). A profile shows
speed and robustness on one picture, and it rewards a solver for being *reliably* close to the best rather than
for winning races; the plot of `plot_profile` is the step version of this table on a logarithmic $\tau$ axis.
Dolan and Moré published the original as MATLAB code; `perf_profile.m` is the Octave version, with two
`%!test` blocks on a table small enough to check by hand.

```python title="performance_profile.py"
--8<-- "docs/part-4-optimization/ch26-experimental-methodology/src/performance_profile.py"
```

```octave title="perf_profile.m"
--8<-- "docs/part-4-optimization/ch26-experimental-methodology/src/perf_profile.m"
```

## Exercise 4

The command, following `docs/benchmark.md` (run from ToOp's root):

```
uv run python -m toop-engine-benchmark.benchmark_toop --multirun grid=config_grid_node_breaker \
    ga_config.random_seed=1,2,3,4,5,6,7,8,9,10 ga_config.runtime_seconds=30,60
```

Twenty runs, in `${grid.output_dir}/multirun/<date>/<time>/<job>` (`benchmark.yaml`), each with its
`benchmark_summary.json`; `assess_benchmarks.py root=...` aggregates them. Checking the keys against
`toop-engine-benchmark/configs/ga_config/ga_config.yaml` (ToOp `ba1874e`): it defines `random_seed` (34523412),
`runtime_seconds` (60), `switching_distance` (20), `n_worst_contingencies` (2) and `substation_split_prob`
(0.1), so both overrides above are valid. The documented example `ga_config.split_subs=2,5` is **not** defined
there, and Hydra rejects an override of a key that is not in the config ("Could not override
'ga_config.split_subs'. Key 'split_subs' is not in struct"; `+ga_config.split_subs=2,5` would add it). Nor does
the script read it: `benchmark_toop.py` builds the `split_subs` descriptor as
`{"metric": "split_subs", "num_cells": cfg.lf_config.max_num_splits}` and takes `switching_distance` from
`cfg.ga_config.switching_distance`, so the number of split substations is swept with
`lf_config.max_num_splits=2,5` (defined in `configs/lf_config/lf_config.yaml`, default 4). Two more details for
whoever fixes the page: the configs live in `toop-engine-benchmark/configs/`, not `config/` as the page says, and
`output_dir` is `${oc.env:PWD}/data/...`, which needs the `PWD` variable that a Windows shell does not set.

## Exercise 5

`rank_tests.power_rank_sum` draws $n$ values from $\mathcal N(0, 3^2)$ and $n$ from $\mathcal N(\delta, 3^2)$,
runs the two-sided Mann–Whitney test and repeats 2 000 times (seeded); the power is the fraction of repetitions
with $p < 0.05$:

| true difference | $n = 5$ | $n = 10$ | $n = 30$ |
|---|---|---|---|
| 0 MW | 0.032 | 0.048 | 0.048 |
| 1 MW | 0.056 | 0.105 | 0.236 |
| 2 MW | 0.099 | 0.267 | 0.715 |
| 4 MW | 0.356 | 0.758 | 0.998 |

The first row is the size of the test: about 5 % false positives as designed, slightly less for $n = 5$ because
the exact rank-sum distribution is discrete and cannot hit 0.05 exactly. A difference of one third of a standard
deviation (1 MW) is hopeless with fewer than a hundred runs; two thirds (2 MW) needs about 30 runs per
configuration for a 70 % chance of being seen; a 4 MW difference, more than one standard deviation, is caught
three times in four with 10 runs and almost surely with 30. The ten seeds of the lab therefore only detect
effects of about one standard deviation of the run-to-run noise, which is why Lab step 4 says "no difference
shown" rather than "no difference". How many to schedule: decide first the smallest difference worth detecting
(say 2 MW, the size of a small redispatch) and the power wanted (80 %), then read the table or extend it: about 40
runs per configuration for 2 MW, 10 for 4 MW. Scheduling fewer and then reporting a non-significant result as
evidence of equality is the mistake Exercise 1 describes from the other side.
