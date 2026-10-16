# Chapter 33 – Solutions

Solutions of the Lab and the Exercises of [Chapter 33](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`dcgrid.py`](src/dcgrid.py) (PTDF, LODF and the pandapower extractor
shared by the other programs), [`planted.py`](src/planted.py) and [`recompile.py`](src/recompile.py) (the Lab in
miniature), [`thresholds118.py`](src/thresholds118.py), [`bridge_debug.py`](src/bridge_debug.py),
[`pydantic_extras.py`](src/pydantic_extras.py), [`bsdf_property.py`](src/bsdf_property.py),
[`lodf_float32.m`](src/lodf_float32.m) (Octave + MATPOWER), and three programs that need ToOp's interpreter,
[`test_bridge_reference_toop.py`](src/test_bridge_reference_toop.py),
[`pydantic_extras_toop.py`](src/pydantic_extras_toop.py) and
[`profile_run_solver_toop.py`](src/profile_run_solver_toop.py). Every number below was printed by one of these
programs (`python <name>.py` prints the numbers its solution quotes); the ToOp numbers come from ToOp commit
`ba1874e` run on the CPU of the maintainer's machine, not in the container. The worked examples run live in
[examples.ipynb](src/examples.ipynb).

## Lab

The lab proper needs an instructor's branch of ToOp with a planted bug, so it is not run here. What follows is
the same loop run on a miniature that is small enough to read: `planted.py` builds IEEE 14 from pandapower
with one transformer shifted by $10^\circ$, computes PTDF, PSDF, LODF and N-1 flows in NumPy, and plants, on
request, the three faults of the instructor's list that can be planted in NumPy, (a) `psdf_sign`, (b) `float32`
and (c) `wrong_slack`; `recompile.py` plants (d), the lambda that is rebuilt on every call. The oracle is
pandapower's own DC power flow of the same grid (`dcgrid.pandapower_dc` reads the internal MATPOWER-style
arrays, so the branch order and the units are the same). The ToOp commands of each step are listed as the method
to follow on the real branch.

1. **Reproduce.** On the ToOp branch: `uv run pytest packages/dc_solver_pkg/tests/jax -q -m "not performance"`,
   then the cells of `notebooks/example1_dc_loadflow_example.ipynb`, writing down every failing test, the
   `overload_energy_n_1` of the unsplit grid, the wall-clock time of the solver calls and every warning (the
   `jax_enable_x64 is set to False` warning of `convert_to_jax.py` is the symptom of bug (b)). On the miniature,
   `planted.symptoms` runs six checks per bug (True means the check passes):

    | Check | correct | `psdf_sign` | `float32` | `wrong_slack` |
    |---|---|---|---|---|
    | oracle, signed flows, atol $10^{-6}$ MW | True | False | False | False |
    | oracle, \|flows\|, atol $10^{-6}$ MW | True | False | False | False |
    | oracle, signed flows, atol $10^{-2}$ MW | True | False | True | False |
    | oracle at zero shift, atol $10^{-6}$ MW | True | True | False | False |
    | slack column of the PTDF is zero | True | True | True | False |
    | LODF N-1 equals re-solve, atol $10^{-8}$ | True | True | False | True |

    The N-1 overload energy (limits 5 % above the correct base-case flows, at least 20 MW) is 520.249 MW for the
    correct code, 594.488 MW with the PSDF sign flipped, 520.249 MW in float32 (no visible change at three
    decimals) and 293.958 MW with the wrong slack. Two lessons already: a wrong answer can look plausible (the
    slack bug lowers the overloads, which a team would be happy to see), and a precision bug has no symptom at
    the level of the metric.

2. **Localize.** On the ToOp branch, compare the arrays `static_information` holds after preprocessing with an
   independent computation of the same grid (your NumPy PTDF of Chapter 17, `numpy_reference.py`, a pandapower
   or pypowsybl DC power flow), in this order: PTDF, then PSDF, then N-0 flows, then N-1 flows; add
   `jax.debug.print` inside the kernel that disagrees and, for a performance symptom, record a profiler trace
   (Exercise 6). `planted.localize` does the array comparison on the miniature (largest absolute difference
   against the correct code):

    | Bug | PTDF | PSDF (from the correct PTDF) | N-0 flows at zero shift | N-0 flows at $10^\circ$ |
    |---|---|---|---|---|
    | `psdf_sign` | 0 | $3.4 \times 10^{2}$ MW/rad | $1.7 \times 10^{-13}$ MW | 59.3 MW |
    | `float32` | $2.9 \times 10^{-8}$ | 0 | $1.3 \times 10^{-6}$ MW | $2.2 \times 10^{-6}$ MW |
    | `wrong_slack` | 0.84 | 0 | 184 MW | 184 MW |

    Each row points at one function: the PSDF columns (the PTDF is untouched and the flows agree as soon as the
    shift is zero), the storage precision (every array moves by about $10^{-8}$, the float32 unit roundoff), the
    PTDF reduction (one whole column is wrong and the error does not depend on the shift).

3. **Test first.** The table of step 1 says which test fails on each branch and which one would have let the bug
   through:
    - (a) A signed oracle test with a non-zero shift angle catches the sign flip; the same oracle on a grid
      whose transformers are at tap zero passes, because the PSDF term vanishes. ToOp's `test_loadflows_match.py`
      compares absolute flows with `rtol=1e-3, atol=1e-3`; absolute values do not matter here (the two
      magnitudes differ, 59 MW on this grid), what matters is that the test grid must have a non-zero shift.
    - (b) A tolerance test justified by the arithmetic catches float32: double precision reproduces pandapower's
      flows to $10^{-9}$ MW, so an absolute tolerance of $10^{-6}$ MW on 150 MW flows (a relative $10^{-8}$,
      a hundred times the double-precision roundoff of a 14-bus solve) fails at the float32 error of
      $2.2 \times 10^{-6}$ MW, while the "engineering" tolerance of $10^{-2}$ MW passes it. ToOp's
      `rtol=1e-3` would pass float32 too. The Hypothesis property of the worked example also fails in float32
      at `atol=1e-8`, which is why the tolerance of a property must be justified like any other.
    - (c) The slack bug is caught by the cheapest possible property, "the slack column of the PTDF is zero", and
      by any oracle whose injections do not sum to zero (pandapower's slack generator absorbs the imbalance,
      the wrong column spreads it elsewhere). The LODF property passes: it compares the buggy code with itself.
    - (d) `recompile.run` counts traces: five calls with a fresh `lambda` trace five times and take about 10 ms
      each on this CPU (the first about 20–30 ms); the module-level function traces once and the following
      calls take about 0.3 ms. The regression test asserts the trace count, not the time, so it does not
      depend on the machine. In ToOp the aggregation function is a `static_argnames` entry of the `jax.jit` in
      `dc_solver/jax/topology_looper.py`, so a new function object is a new cache key.

4. **Fix and report.** The fix is one line in each case (remove the minus sign, keep `jax_enable_x64`, use
   `solver_config.slack`, hoist the lambda to module level, the way `case7_toop.py` of Chapter 19 defines
   `passthrough` once). `test_planted.py` is the regression suite of the miniature: it asserts the whole table
   of step 1, so a fix that changed another row would be noticed, and `pytest --cov` on the branch shows
   whether the changed lines are executed by the new test. The one-page report has four headings, symptom
   (step 1), root cause (step 2), why it escaped (the "misses it" column of step 3) and test added (step 3). The
   unplanted fifth case, the stale limits of `run_solver` from the Chapter 32 lab, follows the same loop; its
   root cause is the hash of `DefaultAggregateOutputFn`, which depends on `solver_config` only.

## Exercise 1

The three denominators have different dimensions. The LODF denominator $1 - (\mathrm{PTDF}_{b,f_b} -
\mathrm{PTDF}_{b,t_b})$ is a difference of PTDF entries, which are ratios of susceptances: multiplying every
reactance by $s$ divides every susceptance by $s$, and the ratio does not move. The MODF matrix $I -
\mathrm{PTDF}_{OO}$ of ToOp's multi-outage path is made of the same entries, so its determinant is invariant too,
although it does scale with the matrix size: $k$ outages with entries around $0.5$ give a determinant around
$0.5^k$, which is why `1e-10` is a different kind of threshold for a $2 \times 2$ and a $3 \times 3$ system. The BSDF
denominator is $\sum_{k \in A} b_k\,(1 - (p_{c,o_k} - p_{c,i}))$, a sum of susceptances times a dimensionless
factor: it is divided by $s$. `thresholds118.py` checks this on IEEE 118 (186 branches, 118 buses) with
pandapower's DC model, one split per bus with four or more branches (36 splits, the first two branches stay)
and the $2 \times 2$ determinant of every pair of consecutive branches:

| Factor $s$ | Kind | Smallest \|denominator\| above threshold | Below threshold | Of |
|---|---|---|---|---|
| 0.1, 1, 10 | LODF | $5.306 \times 10^{-2}$ | 9 | 186 |
| 0.1 | BSDF | $1.312 \times 10^{2}$ | 0 | 36 |
| 1 | BSDF | $1.312 \times 10^{1}$ | 0 | 36 |
| 10 | BSDF | $1.312 \times 10^{0}$ | 0 | 36 |
| 0.1, 1, 10 | determinant | $1.035 \times 10^{-2}$ | 40 | 185 |

The ratios scaled/unscaled are exactly 1 for every LODF denominator and every determinant above threshold, and
exactly $1/s$ (10 and 0.1) for every BSDF denominator. The nine LODF denominators below `1e-11` are the eight
radial branches of IEEE 118 (exactly 0 in double precision) plus one at $10^{-16}$; the 40 determinants below
`1e-10` are the consecutive pairs that island a bus together (the two branches of a degree-two bus, or a pair
containing a radial branch), and all of them stay below the threshold for every $s$. So the LODF and the
determinant decisions cannot change under a uniform rescaling of the reactances, and the BSDF decision can: a
threshold of `1e-5` siemens-per-unit means nothing until the per-unit base is fixed, and on IEEE 118 the
smallest denominator would cross it at $s \approx 10^{6}$, in another unit system much earlier. ToOp does not
document this choice (its denominators are in the per-unit system of the imported grid).

```python title="thresholds118.py"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/thresholds118.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch33-testing-debugging-profiling/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

`bridge_debug.py` uses a five-node ring with a sixth node hung on node 0 through branch 5, the bridge, and a
jitted function that computes all post-contingency flows through the LODF formula without checking the
denominator. The four behaviors:

1. **Plain `jit`.** The call returns a $6 \times 6$ array; only row 5 (the bridge outaged) contains NaN
   (`[nan nan nan nan nan 0.]`): the denominator is $1 - c_5 = 0$, the ring branches have $c_{k,5} = 0$ (the
   pendant node's power only travels through the bridge), so they get $0/0$, and the bridge itself gets the
   forced diagonal $-1$. Nothing is raised; a batched optimizer would carry the NaN into its metric and rank
   the topology by it.
2. **`jax_debug_nans`.** The same call raises `FloatingPointError: invalid value (nan) encountered in div`
   after JAX logs "Invalid nan value encountered in the output of a jax.jit function. Calling the de-optimized
   version": it re-runs the program op by op to find the culprit. Good for a debugging session, unusable in a
   batch of thousands of topologies where one bridge would abort them all. One trap found while testing this:
   an executable compiled while the flag was off carries no check, and JAX 0.11 reused it from the jit cache
   after other jitted functions had run in the same process, so the flag seemed to do nothing. `run_debug_nans`
   therefore calls `jax.clear_caches()` right after setting the flag; do the same in a notebook where the
   function has already run.
3. **`jax.debug.print`.** With `verbose=True` the function prints `LODF denominators: [0.2 0.2 0.2 0.2 0.2 0. ]`
   from inside the compiled program and still returns the NaN; the print survives `jit` because it is a
   callback, and it says exactly where the zero is.
4. **The reference.** `calc_lodf` from ToOp's `numpy_reference.py` raises
   `ValueError: Denominator of LODF is too small - this outage will split the grid` for outage 5 and agrees
   with the jitted flows to $10^{-12}$ for the five other outages (`test_bridge_reference_toop.py`, run with
   ToOp's interpreter).

A batched solver can neither raise (one bad topology in a compiled batch cannot stop the others, and an
exception cannot cross a `jit` boundary anyway) nor return NaN silently (it would poison the metrics). The right
behavior is the fifth one in `bridge_debug.py`, and ToOp's: return the flows and a success flag per outage,
`abs(den) > 1e-11`, here `[True, True, True, True, True, False]`, and let the aggregation mask the failed rows
(ToOp's `lodf.py` does exactly this, with the flag feeding `branches_to_fail`). The reference may raise because
it is a sequential oracle for tests.

```python title="bridge_debug.py"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/bridge_debug.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch33-testing-debugging-profiling/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 3

`PreprocessParameters(enable_bb_outage=True).model_dump()` has no `enable_bb_outage` key and
`preprocess_bb_outages` is still `False`: Pydantic's default `extra='ignore'` accepts and discards the keyword.
The helper `extra_policy(model_cls)` in `pydantic_extras.py` validates the probe `{'no_such_field_42': 1}` and
answers `'forbid'` when the error list contains an `extra_forbidden` entry, `'allow'` when the key survives in
`model_dump()`, and `'ignore'` otherwise (a model with required fields answers with "missing" errors, which is
not a rejection of the unknown key); `assert_forbids_extras` is the pytest helper that fails on anything but
`'forbid'`. `pydantic_extras_toop.py` imports every module of `toop_engine_interfaces.messages` and of
`toop_engine_topology_optimizer.interfaces.messages` and applies the helper to every model defined there. At
`ba1874e` the answer is uniform: the 43 models of `interfaces/messages/` (among them `PreprocessParameters`,
`AreaSettings`, `LimitAdjustmentParameters`, `StartPreprocessingCommand`, the load-flow service commands and
results) and the 27 of `optimizer/interfaces/messages/` (`BatchedMEParameters`, `LoadflowSolverParameters`,
`ACGAParameters`, `StartOptimizationCommand`, `Metrics`, `Topology`, `Strategy`, the heartbeats) all accept
unknown keywords; none forbids or keeps them. `ActionSet` (`interfaces/stored_action_set.py`, outside the
message folders) is the one model that forbids extras. The consequence is the one the chapter describes:
`notebooks/example2_small_grid_toop.ipynb` and `benchmark_toop.py` pass `enable_bb_outage` and
`bb_outage_as_nminus1` to `PreprocessParameters` and run without busbar outages; and
`compute_metrics_single_timestep` loses `worst_k_contingent_cases` because `Metrics` ignores it (Chapter 35,
Exercise 2). Adding `model_config = ConfigDict(extra='forbid')` to a shared base class would turn all of these
into `ValidationError`s at the call site.

```python title="pydantic_extras.py"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/pydantic_extras.py"
```

```python title="pydantic_extras_toop.py"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/pydantic_extras_toop.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch33-testing-debugging-profiling/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`bsdf_property.py` writes the Chapter 18 update in the form ToOp uses, a boolean mask over the branches at the
split node (True = the branch moves to the new busbar, the node's injection stays): with $A$ the branches that
stay, $p_c = \sum_{k \in A, t_k = i} h_k - \sum_{k \in A, f_k = i} h_k + e_i$ and the new column a copy of column
$i$, the updated PTDF is $H + \beta\, p_c^{\mathsf T}$ with $\beta = \mathrm{num}/\mathrm{den}$, $\mathrm{den} =
\sum_{k \in A} b_k\,(1 - (p_{c,o_k} - p_{c,i}))$. The strategy `split_grids` draws a ring with two to five random
chords, a non-slack node with at least four branches and a mask that moves between one and $d-1$ of them. Two
properties, 150 examples each in `test_bsdf_property.py`:

- the updated PTDF equals the PTDF rebuilt from the split network (`rebuilt_ptdf` re-attaches the moved branches
  to a new node and solves again), to $10^{-8}$;
- Kirchhoff's current law holds at both busbars: the net flow leaving the old busbar equals the node's
  injection, the net flow leaving the new one is zero, to $10^{-8}$.

Both pass on the correct code. The off-by-one (`np.roll(mask, 1)`, the mask applied to the wrong branches) is
found at once; `smallest_counterexample` lets `hypothesis.find` shrink it (`derandomize=True`, so the example
is reproducible): 3 buses and 5 branches, the ring 0–1, 1–2, 2–0 plus two parallel chords 0–1, unit susceptances
and zero injections, node 1 split with branch 0 (the first 0–1) kept and the other three moved. The KCL residual
is still 0 (with zero injections every flow is zero, which is why Hypothesis could shrink the injections away),
but the PTDF mismatch is exactly 1.000, a whole unit transfer attributed to the wrong busbar. That is the
report to hand in: the counterexample, the property it violates and the line that caused it.

```python title="bsdf_property.py"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/bsdf_property.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch33-testing-debugging-profiling/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

Octave 11.3 with MATPOWER 8.1. Octave has no single-precision sparse matrices, so `lodf_float32.m` carries a
copy of `makeLODF`'s formula on full matrices (`lodf_from_ptdf`, checked against `makeLODF` on `case9` in the
first test block) and feeds it `full(PTDF)` and `single(full(PTDF))`. On `case118` (186 branches) 9 branches are
radial, with a denominator of exactly 0 in double and in single precision, and their LODF columns are NaN or
Inf in both: float32 does not create or remove a bridge here. On the 177 meshed branches the denominators lie
between $5.306 \times 10^{-2}$ and $0.825$, and the largest float32 error per outage lies between about
$10^{-8}$ and $4.846 \times 10^{-7}$, with a median of $8.1 \times 10^{-8}$; the worst outage is branch 104,
with denominator 0.0805. Plotted on log axes (the code returns the two vectors; `loglog(abs(denom(meshed)),
err(meshed), 'o')` draws it) the points fall below the line $\varepsilon_{32}/|\mathrm{den}|$ with
$\varepsilon_{32} = 1.19 \times 10^{-7}$: the measured ratio error / ($\varepsilon_{32}$ / denominator) lies
between 0.06 and 0.63, so the error grows as the inverse of the denominator, as the quotient rule predicts.
Whether `1e-11` makes sense for float32 data: no. The denominator itself is known to about $\varepsilon_{32}
\approx 10^{-7}$ in float32, so a threshold four orders of magnitude below the noise is the same as testing for
exact zero, and a denominator of $10^{-6}$ (an almost-bridge) passes the check while its LODF column carries an
error of order $\varepsilon_{32}/10^{-6} \approx 0.1$, a 10 % error on every redistributed megawatt. A float32
threshold should sit above the rounding noise, say $10^{-5}$, and the honest choice is to keep
`jax_enable_x64` on, as ToOp does.

```octave title="lodf_float32.m"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/lodf_float32.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch33-testing-debugging-profiling/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 6

`profile_run_solver_toop.py` loads `static_information.hdf5` of `data/case57_data_powsybl` (PTDF $82 \times 72$:
the node-breaker expansion of case57 adds busbars), splits it into `dynamic_information` and `solver_config`,
and calls `run_solver(default_topology(sc, batch_size=4), None, None, di, sc)` twice with
`jax.block_until_ready`, then clears the caches and repeats the two calls under `jax.profiler.start_trace(...,
create_perfetto_trace=True)`. CPU numbers of this machine (JAX 0.5.3, no GPU), which vary by about 30 % between
runs:

- first call, trace + compile + run: 1.2–1.6 s; second call, compiled: 5–9 ms, a ratio between 160 and 300;
- under the profiler: first call 1.8 s, second 7 ms. The trace (`perfetto_trace.json.gz`, read by
  `trace_totals`, which sums only the outermost event of each kind because trace events nest) attributes to the
  first call 0.77 s of tracing (`trace_to_jaxpr_dynamic`, 20 outermost events: `run_solver` traces a tree of
  jitted functions) and 0.156 s of XLA compilation (`backend_compile`, 18 programs: ToOp jits many small
  helpers besides `iterate_symmetric_sequential`); the compiled call itself is the 5–9 ms of the second call.

So in the first call about 40 % is Python tracing, about 10 % is XLA compilation, under 1 % is device
computation and the remaining half is Python overhead around them (building and caching jaxprs, lowering,
dispatch, the profiler's own hooks). In the second call the device computation and the dispatch overhead are
of the same order, a few milliseconds each, which is why batch size matters: the fixed cost per call is paid
once per batch. Perfetto (<https://ui.perfetto.dev>, open `perfetto_trace.json.gz`) shows the same picture as
one long Python-thread bar for the first call with `tf_xla-cpu-llvm-codegen` threads active for a tenth of it,
and a thin bar for the second. Not run here: the ToOp container and py-spy (not installed in ToOp's
environment); `py-spy record -o profile.svg -- python profile_run_solver_toop.py` would show the same split
from the Python side, with the first call's samples under `pjit`/`trace_to_jaxpr` frames and the second
call's under the executable's `__call__`, and inside Docker it needs `--cap-add SYS_PTRACE`, as py-spy's
README and ToOp's `docker/README.md` on the course fork explain.

```python title="profile_run_solver_toop.py"
--8<-- "docs/part-5-computing/ch33-testing-debugging-profiling/src/profile_run_solver_toop.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch33-testing-debugging-profiling/src){ target="_blank" } to see or download all the files of this chapter.
