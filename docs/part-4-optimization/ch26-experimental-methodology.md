# Chapter 26. Experimental methodology for stochastic optimizers

**Weeks:** 1.5 · **Semester:** 5 · **Prerequisites:** [Chapter 5](../part-1-math/ch05-probability.md), [Chapter 25](ch25-quality-diversity.md)

!!! info "Why it matters for ToOp"

    A single ToOp run is one sample from a random process, and two runs with the same configuration are not
    guaranteed to agree. Before you claim that a change "improves the optimizer", you need the tools of this
    chapter.

    - **The budget is wall-clock time.** The DC loop in `optimizer/dc/main.py` runs
      `while time.time() - running_means.start_time < args.ga_config.runtime_seconds`, and the Kafka workers stop
      the same way (`optimizer/dc/worker/worker.py`, `optimizer/dc_bruteforce/worker.py`, `optimizer/ac/worker.py`).
      The number of completed epochs (each `iterations_per_epoch` iterations, default 100) therefore depends on
      the hardware, the load on the machine and the JIT compilation of the first epoch.
    - **A seed is not enough.** `ga_config.random_seed` (default 42) seeds the JAX PRNG key
      (`optimizer/dc/genetic_functions/initialization.py`), so the sequence of epochs is repeatable, but a faster
      or slower machine stops that sequence at a different epoch and reports a different result.
    - **Inputs can differ across platforms.** Preprocessing runs native code (pypowsybl's load flow, BLAS), and
      nothing guarantees bit-identical results on every operating system. Before comparing runs from two
      machines, compare their inputs: the unsplit `overload_energy_n0` and `overload_energy_n1` that `load_grid`
      writes to `static_information_stats.json` for the same case57 grid on each machine.
    - **What ToOp records.** After every epoch `optimizer/dc/main.py` logs the best fitness and its metrics to
      TensorBoard (`log_tensorboard`) and writes `res_<epoch>.json`, whose top level includes the cumulative
      evaluation counters `total_branch_combis`, `total_inj_combis` and `total_num_splits`
      (`MixingEmitterState` in `optimizer/dc/ga_helpers.py`). Those files are the raw material for anytime curves.
    - **Tuning tools.** `docs/benchmark.md` describes Hydra multiruns with `toop-engine-benchmark/benchmark_toop.py`
      and an assessment script, `toop-engine-benchmark/assess_benchmarks.py`. Optuna is declared as a dependency
      in `packages/topology_optimizer_pkg/pyproject.toml`, but no ToOp module imports it.

## Topics

- Why stochastic optimizers need repeated runs: run-to-run variance, lucky seeds, reporting the best of $n$.
- Budgets: fixed number of evaluations vs fixed wall-clock time vs fixed target; fair comparison when the cost per
  evaluation differs; the JIT warm-up epoch.
- Reproducibility: seeds and PRNG streams, recording configuration and versions (`res.json` stores the full
  `args`), hardware and platform differences, floating-point non-determinism.
- Anytime performance: convergence curves over evaluations and time, median and quantile bands, area under the
  curve.
- Empirical cumulative distribution functions (ECDFs) of final values and of runtime-to-target; COCO-style
  runtime ECDFs aggregated over targets.
- Performance profiles (Dolan and Moré) for comparing several solvers over many instances.
- Statistical testing: null hypotheses, $p$-values, the Wilcoxon rank-sum (Mann–Whitney U) test for independent
  runs, the Wilcoxon signed-rank test for paired instances, the Friedman test with post-hoc comparisons for
  several algorithms, multiple-testing correction (Holm).
- Effect sizes and confidence: Vargha–Delaney $\hat A_{12}$, bootstrap confidence intervals of the median;
  statistical vs practical significance.
- Hyperparameter tuning: grid and random search, Hydra multirun sweeps, Optuna, racing (irace); tuning and test
  instances must be separate.
- Ablation studies: switching off one component (crossover, a descriptor, a mutation type) at a time.
- Reporting: what to publish (configuration, seeds, budget, hardware, all runs, not only the best), and the
  indicator statistics used for Pareto fronts in [Chapter 27](ch27-pareto-multi-objective.md).

## Learning outcomes

- Design a comparison of two optimizer configurations with a fixed evaluation budget, repeated seeds and a
  pre-registered metric, and explain why a wall-clock budget confounds the result.
- Plot anytime curves with quantile bands and ECDFs of final quality from repeated runs.
- Choose and run the right nonparametric test (rank-sum, signed-rank, Friedman), report an effect size, and state
  the conclusion without over-claiming.
- Explain which of ToOp's outputs make a run reproducible and which do not.

## Resources

- Papers:
  - Bartz-Beielstein, Doerr, van den Berg, Bossek et al., "Benchmarking in Optimization: Best Practice and Open
    Issues", arXiv:2007.03488, 2020. **Main reading**: sections on goals, problem instances, algorithms,
    performance measures, analysis and presentation, and reproducibility.
  - Derrac, García, Molina & Herrera, "A practical tutorial on the use of nonparametric statistical tests as a
    methodology for comparing evolutionary and swarm intelligence algorithms", *Swarm and Evolutionary
    Computation* 1(1):3–18, 2011, doi:10.1016/j.swevo.2011.02.002.
  - Dolan & Moré, "Benchmarking optimization software with performance profiles", *Mathematical Programming*
    91(2):201–213, 2002, doi:10.1007/s101070100263.
  - Hansen, Auger, Ros, Mersmann, Tušar & Brockhoff, "COCO: A Platform for Comparing Continuous Optimizers in a
    Black-Box Setting", arXiv:1603.08785; platform <https://numbbo.github.io/coco/>.
  - Hansen, Auger, Brockhoff & Tušar, "Anytime Performance Assessment in Blackbox Optimization Benchmarking",
    *IEEE TEC* 26(6):1293–1305, 2022, doi:10.1109/TEVC.2022.3210897 (runtime ECDFs).
  - Arcuri & Briand, "A Hitchhiker's guide to statistical tests for assessing randomized algorithms in software
    engineering", *Software Testing, Verification and Reliability* 24(3):219–250, 2014,
    doi:10.1002/stvr.1486 (number of runs, rank-sum test, $\hat A_{12}$).
  - Vargha & Delaney, "A Critique and Improvement of the CL Common Language Effect Size Statistics of McGraw and
    Wong", *Journal of Educational and Behavioral Statistics* 25(2):101–132, 2000,
    doi:10.3102/10769986025002101.
  - López-Ibáñez, Dubois-Lacoste, Pérez Cáceres, Birattari & Stützle, "The irace package: Iterated racing for
    automatic algorithm configuration", *Operations Research Perspectives* 3:43–58, 2016,
    doi:10.1016/j.orp.2016.09.002; package <https://cran.r-project.org/package=irace>.
- Documentation:
  - SciPy: `scipy.stats.mannwhitneyu`
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html>, `wilcoxon`
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html>, `friedmanchisquare`
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html>, `ecdf`
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html>.
  - Hydra, multi-run <https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/>; Optuna
    <https://optuna.readthedocs.io/en/stable/>; TensorBoard <https://www.tensorflow.org/tensorboard>.
  - ToOp: `docs/benchmark.md`, `toop-engine-benchmark/benchmark_toop.py`, `toop-engine-benchmark/configs/`.
- Textbook: none of the surveyed syllabi teaches this topic; the probability background is
  [Chapter 5](../part-1-math/ch05-probability.md).

## Worked example

Ten independent runs per configuration on one instance are independent samples, so use the rank-sum test, and
report an effect size next to the $p$-value:

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
a = rng.normal(42.0, 3.0, size=10)   # final overload energy [MW] of 10 runs, configuration A
b = rng.normal(39.0, 3.0, size=10)   # configuration B (lower is better)

res = stats.mannwhitneyu(a, b, alternative="two-sided")
# Vargha-Delaney A12: probability that a random run of A beats (is lower than) a random run of B
a12 = (np.sum(a[:, None] < b[None, :]) + 0.5 * np.sum(a[:, None] == b[None, :])) / (a.size * b.size)
print(f"U={res.statistic:.1f}  p={res.pvalue:.4f}  A12={a12:.2f}")   # U=92.0  p=0.0017  A12=0.08

for x, label in ((a, "A"), (b, "B")):
    stats.ecdf(x).cdf.plot(plt.gca(), label=label)
plt.xlabel("final overload energy [MW]"); plt.ylabel("fraction of runs"); plt.legend(); plt.show()
```

$\hat A_{12} = 0.08$ means a run of A beats a run of B only 8% of the time. Use `stats.wilcoxon` (signed-rank)
instead when the samples are paired, for example the same seed on the same instance for both configurations
across several different instances.

## Lab

!!! example "Lab: 10 seeds, 2 configurations, one honest conclusion"

    **Tools:** Jupyter with NumPy, pandas, matplotlib and SciPy; your GA or MAP-Elites code from
    [Chapter 24](ch24-evolutionary-algorithms.md) and [Chapter 25](ch25-quality-diversity.md). Optional: ToOp.

    1. **Protocol first.** Write down, before running anything: the instance (the IEEE 30 line-switching problem
       of Chapter 24), the two configurations (for example with and without crossover, or two mutation rates),
       the budget (a fixed number of fitness evaluations), the performance measure (best overload energy at the
       end of the budget), 10 seeds per configuration, the test and the significance level.
    2. **Run.** Log best-so-far fitness after every evaluation batch together with the evaluation count and the
       elapsed time; store one pandas row per (configuration, seed, evaluation).
    3. **Anytime curves.** Plot the median and the 25–75% quantile band of best-so-far value against evaluations
       and, separately, against seconds.
    4. **ECDFs and tests.** Plot the ECDF of final values per configuration, run `scipy.stats.mannwhitneyu`,
       compute $\hat A_{12}$ and a bootstrap 95% confidence interval of the difference in medians. Write the
       conclusion in two sentences.
    5. **Wall-clock trap.** Repeat the comparison with a wall-clock budget while another heavy process runs on the
       machine for half of the runs. Show how the conclusion changes.
    6. **Optional, ToOp.** Run `notebooks/example2_small_grid_toop.ipynb` with 5 values of
       `ga_config.random_seed` and the same `runtime_seconds`. Collect `max_fitness`, `iteration` and
       `total_branch_combis` from every `res_<epoch>.json`, plot the anytime curves against epochs and against
       `total_branch_combis`, and truncate all runs to the smallest common epoch to compare them under an equal
       budget.

## Exercises

1. A paper reports "our method found the best topology in 1 of 20 runs; the baseline never did" and compares only
   the best run of each method. List three things wrong with this report and propose a better table.
   *(pen and paper)*
2. Three configurations are run on 8 grid instances (5 seeds each, median taken per instance). Apply the Friedman
   test with `scipy.stats.friedmanchisquare`, then Holm-corrected pairwise signed-rank tests, on synthetic data
   where one configuration is slightly better. *(Jupyter: NumPy + SciPy)*
3. Build a performance profile $\rho_s(\tau)$ for three solvers over 20 instances from a table of times to reach a
   target, including failures, and interpret $\rho_s(1)$ and $\rho_s(\tau)$ for large $\tau$.
   *(Jupyter: NumPy + pandas + matplotlib)*
4. Using ToOp's `docs/benchmark.md`, write a Hydra multirun command that sweeps `ga_config.random_seed` over 10
   values and `ga_config.runtime_seconds` over two values with `toop-engine-benchmark/benchmark_toop.py`.
   Check every key against `toop-engine-benchmark/configs/ga_config/ga_config.yaml`: is the documented
   `ga_config.split_subs` override defined there, and does the script read it? *(pen and paper)*
5. Simulate the power of the rank-sum test: for true median differences of 0, 1, 2 and 4 MW with standard
   deviation 3 MW, estimate by Monte Carlo the probability of $p < 0.05$ with 5, 10 and 30 runs per
   configuration. How many runs would you schedule? *(Jupyter: NumPy + SciPy + matplotlib)*
