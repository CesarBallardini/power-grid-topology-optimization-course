# Chapter 32 – Solutions

Solutions of the Lab and the Exercises of [Chapter 32](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`solver_lab_toop.py`](src/solver_lab_toop.py) and
[`split_slots_toop.py`](src/split_slots_toop.py) run inside ToOp's environment (the container, or
`../ToOp/.venv`), [`stale_hash.py`](src/stale_hash.py), [`traced_loop.py`](src/traced_loop.py),
[`int_widths.py`](src/int_widths.py) and [`pmap_donate.py`](src/pmap_donate.py) run anywhere; each has a test
next to it (`test_*_toop.py` for the ToOp ones, run with `make test-toop`). Every script prints the numbers quoted
below when run from the `src` folder. The ToOp numbers were produced with ToOp commit `ba1874e` on **CPU**
(JAX 0.5.3, Python 3.11, Windows). Lab steps 2 and 4 were then measured again on CPU *and* GPU in one container
with [`gpu_compare_toop.py`](src/gpu_compare_toop.py) (environment below); the GPU behaviour of Exercise 5 is
described, not measured. The other scripts ran in the course environment (JAX 0.11, NumPy 2.4, Python 3.14) and
again with ToOp's interpreter where the versions matter.

**GPU environment.** An NVIDIA GeForce RTX 3050 Laptop GPU (4 096 MiB, 35 W power limit), Windows driver 617.42,
reached through Docker Desktop's WSL2 backend (`docker run --gpus all`). The image is `toop-toop-gpu:latest`, built
from the course fork's `docker/Dockerfile` with `INSTALL_CUDA=true`, which adds `jax[cuda12]` at the JAX version
of `uv.lock`. ToOp `ba1874e` is bind-mounted read-only at `/app`, and the environment in `/opt/venv` (a named
volume) is synced to that commit's `uv.lock` with `uv sync --frozen --inexact`, which keeps the CUDA wheels:
JAX, jaxlib and `jax-cuda12-plugin` 0.5.3, CUDA runtime 12.9.79, cuDNN 9.24, Python 3.11.16
(`python gpu_compare_toop.py env` prints them). CPU and GPU runs use this same container and differ only in
`JAX_PLATFORMS`: `cpu`, or `cuda,cpu` for the GPU (ToOp's preprocessing also asks for the CPU device, so `cuda`
alone fails with "Unknown backend cpu"). Every run had `--cpus 4`, `XLA_PYTHON_CLIENT_PREALLOCATE=false` and an
otherwise idle machine, and every measurement starts a fresh Python process: a compilation happens once per
process, and JAX's peak-memory counter cannot be reset.

## Lab

1. **Setup.** `solver_lab_toop.setup(folder)` writes `case57_data_powsybl` into the folder the first time, calls
   `load_grid(DirFileSystem(folder), pandapower=False, parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False,
   fail_on_non_convergence=False))` and `run_initial_loadflow`, and returns the `StaticInformation`. The
   preprocessed case57 has 72 nodes (57 buses plus a second busbar for each substation that can be split),
   82 branches, 12 relevant substations (ToOp drops the ones with fewer than four non-bridge branches), 84 N-1
   cases (81 branch outages and 3 generator outages), `number_most_affected = 30` and `batch_size_bsdf = 16`.
   `default_topology(sc, batch_size=4)` is an `ActionIndexComputations` whose `action` array has shape `(4, 1)`:
   one split slot, filled with the padding sentinel. `timed()` wraps every call in `jax.block_until_ready`;
   without it the timer stops when the call is *dispatched*, not when it is done.
2. **Compile vs run.** CPU, JAX 0.5.3, four unsplit topologies:

    | Call | Time |
    |---|---|
    | `run_solver`, call 1 | 1 050 ms (compiles) |
    | `run_solver`, calls 2 and 3 | 6.7 ms, 6.4 ms (cached) |
    | `run_solver_symmetric`, `passthrough` defined once, call 1 | 634 ms (compiles) |
    | `run_solver_symmetric`, `passthrough`, calls 2 and 3 | 3.9 ms, 3.7 ms (cached) |
    | `run_solver_symmetric`, fresh `lambda`, calls 1, 2, 3 | 574 ms, 579 ms, 613 ms (compiles every time) |

    The first call pays for tracing and XLA compilation, about 150 times the cost of running the compiled
    program on this grid. `run_solver` builds a new `DefaultAggregateOutputFn` on every call, yet hits the
    cache, because that class defines `__hash__` as the `fixed_hash` it was given, `hash(solver_config)`, and
    `__eq__` as equality of hashes: two instances built from the same `sc` are the same cache key. A `lambda`
    expression evaluates to a new function object every time, function objects hash by identity, and
    `aggregate_output_fn` is in `static_argnames`, so each call is a new key and a new compilation. The
    pass-through function defined once at module level behaves like `DefaultAggregateOutputFn`. One practical
    note: under JAX 0.5.3, `jax.no_tracing()` cannot wrap `run_solver`, because the Python glue around the
    jitted core (`pad_topologies_action_index`, `prepare_result_storage`, the final slicing) dispatches eager
    primitives, which that version reports as "re-tracing function convert_element_type". The scripts count
    the compiled programs of the core instead, `iterate_symmetric_sequential._cache_size()`, which answers the
    same question exactly; `jax.no_tracing()` works around the jitted core itself, which is how
    `packages/topology_optimizer_pkg/tests/test_jax_compilation.py` uses it.

    *CPU against GPU.* `python gpu_compare_toop.py compile ROOT 5` runs `compile_vs_run` in one discarded
    warm-up process and then in five fresh processes; medians, same container, `JAX_PLATFORMS` switched:

    | Call | CPU | GPU |
    |---|---|---|
    | `run_solver`, call 1 | 1 209 ms | 2 337 ms |
    | `run_solver`, calls 2 and 3 | 6.6 ms, 7.4 ms | 19.0 ms, 17.0 ms |
    | `run_solver_symmetric`, `passthrough`, call 1 | 777 ms | 1 469 ms |
    | `run_solver_symmetric`, `passthrough`, calls 2 and 3 | 4.5 ms, 4.3 ms | 13.7 ms, 13.7 ms |
    | `run_solver_symmetric`, fresh `lambda`, calls 1, 2, 3 | 704 ms, 766 ms, 719 ms | 1 307 ms, 1 303 ms, 1 291 ms |

    The pattern is the same on both devices, because the cache keys do not depend on the device. The GPU
    compiles about twice as slowly (XLA's GPU pipeline does more work and emits device code), and on four
    case57 topologies a cached call is 2.3–3.2 times *slower* than on the CPU: a job this small is all kernel
    launches and host–device copies. The GPU only pays off with large batches (step 4).

    ```python title="solver_lab_toop.py (Lab 2)"
    --8<-- "docs/part-5-computing/ch32-jax-in-toop/src/solver_lab_toop.py:compile"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch32-jax-in-toop/src){ target="_blank" } to see or download all the files of this chapter.

3. **Find the trigger.** `pf_n_1_max` has shape `(n_topologies, n_timesteps, number_most_affected)` and holds
   relative loadings (flow over `max_mw_flow`), sorted; the baseline maximum is 18.6192 (this test grid has
   tiny operational limits). "Compiled" below means the core gained a cache entry:

    | Change | Prediction | Measured | `pf_n_1_max` |
    |---|---|---|---|
    | same `sc`, second call | cached | cached, 6.6 ms | `(4, 1, 30)`, max 18.6192 |
    | `replace(sc)`, no field changed | recompiles | compiled, 730 ms | unchanged |
    | `replace(sc, number_most_affected=20)` | recompiles | compiled, 756 ms | `(4, 1, 20)`, max 18.6192 |
    | `sc2 = replace(sc)`, warm-up, then `sc2.number_most_affected = 20` in place | reuses incorrectly | cached, 6.5 ms | `(4, 1, 30)`: the old 30 columns |
    | `nodal_injections * 1.1` only | reuses, flows unchanged | cached, 7.0 ms | max 18.6192, unchanged |
    | `nodal_injections` and `unsplit_flow * 1.1` | reuses, flows × 1.1 | cached, 6.8 ms | max 18.7222 (see below) |
    | `max_mw_flow / 2`, same `sc` | reuses incorrectly | cached, 7.0 ms | max 18.6192: the old limits |
    | `max_mw_flow / 2`, `replace(sc)` | recompiles | compiled, 670 ms | max 37.2385 = 2 × 18.6192 |
    | batch of 8 topologies | recompiles? | cached, 111 ms | `(8, 1, 30)`, max 18.6192 |

    Every line follows from three facts. (i) `iterate_symmetric_sequential` is jitted with
    `static_argnames=("solver_config", "aggregate_output_fn")`, so those two objects are cache keys and
    everything in `DynamicInformation` is traced data. (ii) `SolverConfig.__hash__` is `id(self)`: `replace`
    makes a new object (new key, recompile, even with identical values), while an in-place assignment keeps the
    key and the stale program, which still writes 30 columns although `sc2.number_most_affected` says 20. (iii)
    `run_solver` builds `DefaultAggregateOutputFn` with `max_mw_flow=dynamic_information.branch_limits.max_mw_flow`
    stored as a Python attribute and `fixed_hash=hash(solver_config)`: inside the jitted core the function is
    static, so its `max_mw_flow` array is baked into the compiled program as a constant. Halving the limits in
    `DynamicInformation` with the same `sc` therefore changes nothing (the hash is the same, the constant is the
    old array), and `replace(sc)` forces a recompile that bakes in the new limits and doubles every relative
    loading. This is exactly the chapter's statement for commit `ba1874e`. Two details worth seeing: scaling
    `nodal_injections` alone does nothing because, with relevant substations present, the N-0 flows come from
    `dynamic_information.unsplit_flow` (passed to `compute_cross_coupler_flows` in
    `dc_solver/jax/compute_batch.py`), not from a fresh `ptdf @ nodal_injections`; scaling both multiplies the
    N-0 maxima by exactly 1.1 and the branch-outage N-1 entries too, but the three generator-outage entries by
    less (their delta P, `nonrel_injection_outage_deltap`, was not scaled), so the worst entry only moves from
    18.6192 to 18.7222. And the batch of 8 does *not* recompile, because `run_solver_symmetric` pads every batch
    to a multiple of `batch_size_bsdf = 16`: 4 and 8 topologies reach the core with identical shapes; the 111 ms
    are eager glue (padding and storage for the new size), not compilation. With `batch_size_bsdf = 4` it would
    recompile.

    *Bug report.* **Title:** `run_solver` ignores a changed `branch_limits.max_mw_flow` unless `SolverConfig`
    is replaced. **Version:** ToOp `ba1874e`, JAX 0.5.3, CPU. **Steps:** load `case57_data_powsybl`;
    `base = run_solver(topologies, None, None, di, sc)`; `di2 = eqx.tree_at(lambda d: d.branch_limits.max_mw_flow,
    di, di.branch_limits.max_mw_flow / 2)`; `again = run_solver(topologies, None, None, di2, sc)`.
    **Expected:** `again.n_1_results.pf_n_1_max == 2 * base.n_1_results.pf_n_1_max`. **Observed:** identical to
    `base` (18.6192); with `replace(sc)` instead of `sc` the result is 37.2385 as expected. **Cause:**
    `DefaultAggregateOutputFn` (`dc_solver/jax/topology_looper.py`) stores `max_mw_flow` as an attribute of a
    static argument and hashes by `hash(solver_config)`, so the compiled program keeps the first limits it saw.
    **Fix options:** pass `max_mw_flow` through `lf_result`/the traced arguments instead of the static object,
    or include a hash of the limits (and of `branches_to_fail`) in `fixed_hash`.

    ```python title="solver_lab_toop.py (Lab 3)"
    --8<-- "docs/part-5-computing/ch32-jax-in-toop/src/solver_lab_toop.py:trigger"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch32-jax-in-toop/src){ target="_blank" } to see or download all the files of this chapter.

4. **Batch size and memory.** `case300_powsybl` (first fifty bus groups relevant, 14 of them survive the
   four-branch filter; 411 branches, 385 N-1 cases). `batch_sweep` runs 64 unsplit topologies for every
   `(batch_size_bsdf, batch_size_injection)` pair, records the first call minus the steady state as compile
   time, the steady-state throughput, and, as a CPU stand-in for `docker stats`, the growth of the process's
   resident memory across the point (psutil). Two runs, CPU:

    | `batch_size_bsdf` | `batch_size_injection` | compile [s] | topologies / s | RSS growth [MB] |
    |---|---|---|---|---|
    | 4 | 1 / 2 / 4 | 1.1–1.3 / 0.8–0.9 / 0.8–0.9 | 513–676 / 546–659 / 582–633 | 65–68 / 41 / 44–46 |
    | 8 | 1 / 2 / 4 | 0.8–0.9 / 0.8 / 0.8 | 470–548 / 466–487 / 466 | 48–54 / 43–56 / 40 |
    | 16 | 1 / 2 / 4 | 0.9 / 1.2 / 1.0 | 472 / 459 / 470 | 75 / 52 / 48 |
    | 32 | 1 / 2 / 4 | 1.0 / 0.9 / 0.9 | 460 / 468 / 468 | 91 / 66 / 55 |

    On a CPU the heat map is flat: 460–680 topologies per second everywhere, slightly *better* for the smallest
    BSDF batch (the intermediate PTDF/LODF blocks of a small batch fit the cache), and the injection batch
    does not matter with one timestep. Memory grows with `batch_size_bsdf` (a PTDF and an LODF block per
    topology in the batch), which is the knob the chapter says to cut first on an out-of-memory error, and
    `batch_size_injection` next. Compile time is 0.8–1.3 s for every point, so a sweep costs more in
    compilation than in running.

    *CPU against GPU.* `gpu_compare_toop.sweep` repeats the sweep in the GPU environment above, one fresh process
    per point, with 1 024 unsplit topologies (or `batch_size_bsdf` of them when that is larger), one compiling
    call, one warm-up call and the median of five. Memory is JAX's peak of bytes in use on the GPU
    (`memory_stats()`, checked against `nvidia-smi`, which adds the CUDA context of the whole card) and the peak
    growth of the process's resident set on the CPU, sampled every 20 ms (reading it after the calls, as above,
    misses the temporaries XLA has already freed). `batch_size_injection` (1, 4, 16 and 64) changes nothing on
    either device with one timestep: GPU throughput stays within 2 % across it. The rows below are
    `batch_size_injection = 1`:

    | `batch_size_bsdf` | CPU float64 [topologies/s] | GPU float64 [topologies/s] | CPU peak RSS growth | GPU peak memory | GPU float32 [topologies/s] | GPU float32 peak |
    |---|---|---|---|---|---|---|
    | 4 | 372 | 4 290 | 76 MB | 22 MB | 12 652 | 8 MB |
    | 16 | 285 | 5 026 | 99 MB | 39 MB | 16 380 | 21 MB |
    | 64 | 270 | 5 130 | 226 MB | 140 MB | 16 702 | 71 MB |
    | 256 | 352 | 5 235 | 721 MB | 389 MB | 19 925 | 272 MB |
    | 1 024 | 283 | 5 306 | 2 694 MB | 1 535 MB | 19 839 | 672 MB |
    | 2 048 | 260 | 5 392 | 5 345 MB | 3 069 MB | 22 663 | 1 343 MB |
    | 4 096 | 237 | out of memory | 10 639 MB | – | 23 286 | 2 685 MB |
    | 8 192 | – | – | – | – | out of memory | – |

    In float64 the GPU is 11.5 times faster than the four CPU cores at `batch_size_bsdf = 4` and 19–21 times
    faster from 1 024 up. Its throughput is already high at 4 and rises only by a quarter up to 2 048: the batch
    of 1 024 topologies is processed in a `fori_loop` over BSDF batches, and even a batch of 4 keeps this
    small card busy. Compilation takes 2.3–3.0 s per point on the GPU and 1.4–2.1 s on the CPU. Memory grows
    linearly, about 1.5 MB per topology in the BSDF batch in float64 (0.66 MB in float32) on the GPU and
    2.6 MB on the CPU, so `batch_size_bsdf` is indeed the knob that sets the footprint. The out-of-memory
    boundary is between 2 048 and 4 096 in float64: 4 096 fails with `RESOURCE_EXHAUSTED: Out of memory while
    trying to allocate 6117278032 bytes`, a single 6.1 GB buffer on a 4 GB card. The limit is not the card but
    JAX's allocator: even with preallocation off it stops at 75 % of the memory (`bytes_limit` 3 221 MB), and
    2 048 with 16 384 topologies peaked at 3 216 MB, right at that edge. `XLA_PYTHON_CLIENT_MEM_FRACTION`
    moves the limit. `nvidia-smi` showed 2 138 MiB at 1 024 and 3 148 MiB at 2 048, the JAX peak plus the
    CUDA context (90–160 MiB at rest). In float32 everything halves, and the boundary moves to between 4 096
    and 8 192. A laptop GPU throttles: every float64 point drew the full 35 W, with the software power cap in
    `nvidia-smi`'s clock-event reasons, mostly at 1.4–1.55 GHz of a 2.1 GHz maximum and 52–60 °C. That is why each
    point has a warm-up call and a median.

    *The float64 penalty.* ToOp switches `jax_enable_x64` on when `convert_to_jax` is imported, so
    `JAX_ENABLE_X64=0` alone does not give a float32 run. `gpu_compare_toop.to_float32` narrows the static
    information and the topologies (float64 to float32, int64 to int32, with the int64 padding sentinel mapped
    to int32's), and then switches x64 off before the first solver call (`sweep ROOT float32`). On the GPU,
    float32 is 2.9 times faster at `batch_size_bsdf = 4` and 3.7–4.2 times faster from 1 024 up. That is far from
    the 1/32 float64 rate of a GeForce card, because the solver moves more data than it multiplies. On the CPU
    the ratio is 1.6–2.0. Speed is not the reason ToOp stays in float64, though. Take 64 random case300
    topologies, each splitting up to two of the 14 relevant substations (`python gpu_compare_toop.py agree
    ROOT`). Float32 flags only 4 of the 12 outages that island a split grid. The other 8 come back as
    successful, with flows up to 1 472.5 MW computed from a denominator that is rounding noise: plausible
    numbers that the optimizer would trust. On the 59 topologies whose outages all succeed, float32 is close:
    3.2e-3 MW on the N-0 flows (2.5e-6 of the largest, 1 294.5 MW), 1.2e-2 MW on the N-1 flows (5.7e-6) and
    5.7e-6 relative on `pf_n_1_max`. On case57 (`agree ROOT case57 8`), one of 8 random topologies (actions
    `[54, 49]`) cannot be solved in float32 at all: its N-0 flows come back infinite and all 84 outages fail.
    In float64 it solves, but 3 of its outages island the grid, so ToOp rejects it.

    *Agreement of the devices.* The same topologies in float64 on the CPU and the GPU, in one process. On
    case300 the success flags are identical: the same 12 islanding outages, in 5 topologies. On the 59
    topologies ToOp accepts, the largest differences are 4.1e-12 MW on the N-0 flows (3.2e-15 relative),
    2.1e-11 MW on the N-1 flows (1.0e-14) and 1.4e-17 on `pf_n_1_max` (6.8e-16). The 5 rejected topologies
    differ by 1.5e-12 MW (N-0) and 7.2e-12 MW (their valid N-1 outages). These differences are rounding, from a
    different order of summation in the GPU kernels. The rows of an islanding outage are meaningless on both
    devices (a division by a near-zero denominator), and they differ. The case57 topology `[54, 49]` shows
    what ill-conditioning does even in float64: CPU and GPU differ by 0.21 MW on its N-0 flows and 0.24 MW on
    its valid N-1 flows, while its 7 accepted neighbours agree to 7e-13 MW. Compare results only where ToOp's
    `success` is True, and treat a topology that it rejects as unsolved, not as nearly solved.

    ```python title="gpu_compare_toop.py (one sweep point)"
    --8<-- "docs/part-5-computing/ch32-jax-in-toop/src/gpu_compare_toop.py:point"
    ```

    ```python title="gpu_compare_toop.py (CPU against GPU, float32 against float64)"
    --8<-- "docs/part-5-computing/ch32-jax-in-toop/src/gpu_compare_toop.py:agree"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch32-jax-in-toop/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

`SolverConfig` holds, besides scalars, `HashableArrayWrapper` fields such as `branches_per_sub` and
`rel_stat_map`, arrays with one entry per relevant substation. A value hash would have to hash those arrays
(and compare them on every cache lookup, since `__eq__` runs after a hash match), on *every* call of
`iterate_symmetric_sequential`, which the optimizer calls millions of times per run; `id(self)` costs nothing
and, as the docstring says, "we expect only one instance of this class to be used at a time". That is the
benefit. The risks: (1) **stale programs**: mutating a field in place keeps the key, so the compiled program
ignores the new value silently (Lab 3: `number_most_affected = 20` still returned 30 columns), and the same
mechanism leaks into `DefaultAggregateOutputFn`, whose `fixed_hash = hash(solver_config)` lets a changed
`max_mw_flow` go unnoticed; (2) **spurious recompilations**: every `replace(...)`, every `load_grid` and every
deserialization produces a new object, so two configurations with identical values never share a program
(Lab 3: `replace(sc)` with no field changed cost 730 ms), which also makes the compile cache grow with every
copy that stays alive; a third risk is that `id()` is recycled by the garbage collector, so a *new* object
may reuse the id of a dead one and hit its cache entry with different values. A safer design: a frozen
dataclass (`@dataclass(frozen=True)`, so in-place mutation is a `FrozenInstanceError`) whose `__hash__`
combines the scalar fields with a hash of the array contents computed once in `__post_init__` (for example
`hash(array.tobytes())` stored in a field, with `__eq__` comparing scalars and `np.array_equal` on the
arrays). It costs one hash computation per object instead of per call, equal configurations share a
program, and mutation is impossible instead of silent. Exercise 2 shows both designs side by side.

## Exercise 2

```python title="stale_hash.py (the two designs)"
--8<-- "docs/part-5-computing/ch32-jax-in-toop/src/stale_hash.py:configs"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch32-jax-in-toop/src){ target="_blank" } to see or download all the files of this chapter.

`worst = jax.jit(_worst, static_argnums=1)` returns the `config.n_worst` largest |flows|. With
`IdentityConfig(n_worst=3)` the result has shape `(3,)`; after `config.n_worst = 5` in place it is still
`(3,)`, the stale program, and the Python counter inside `_worst` confirms that the second call did not trace;
`worst(flows, replace(config, n_worst=5))` traces again and returns `(5,)`. With `FrozenConfig`, 3 → `(3,)`,
5 → `(5,)`, and a third call with a new `FrozenConfig(3)` is a cache hit: two traces in total. The test that
would have caught the trap is the check that a result follows the configuration it was computed with,
`assert worst(flows, config).shape == (config.n_worst,)` after the mutation: `test_stale_hash.py` shows it
failing for `IdentityConfig` (wrapped in `pytest.raises(AssertionError)`, so the suite stays green) and passing
for `FrozenConfig`. In ToOp's own suite, `test_jax_compilation.py` guards the other direction (no spurious
retrace) with `jax.no_tracing()`; a test of this direction (a changed static field must change the output) is
what was missing.

## Exercise 3

```python title="traced_loop.py (the two loops)"
--8<-- "docs/part-5-computing/ch32-jax-in-toop/src/traced_loop.py:loops"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch32-jax-in-toop/src){ target="_blank" } to see or download all the files of this chapter.

Twenty Poisson draws clipped to `[1, 8]` (`[1, 1, 4, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 4, 3, 1, 1]`,
three distinct values) give **1 trace** for the `fori_loop` version and **3 traces** for the Python `for`,
one per distinct bound, and the same sums. `jax.make_jaxpr` shows why: the `fori_loop` with a traced bound
lowers to a single `while` primitive whose `cond_jaxpr` compares the counter with the bound (an *input* of the
jaxpr, `b:i64[]`), so the bound is data; the Python `for` is unrolled at trace time into three copies of the
body (three `dynamic_slice` reads and three `add`s for `n_steps = 3`), so the bound is part of the program and
a new value is a new program. Passing a traced bound to the Python version raises
`TracerIntegerConversionError` (`range()` asks the tracer for an integer it does not have). `mutate.py` is in
the first camp: `n_subs_mutated = jnp.clip(jax.random.poisson(...), 1, max_num_splits)` is a tracer and
`jax.lax.fori_loop(0, n_subs_mutated, ...)` compiles once for every draw; `max_num_splits` (the array width)
stays static and bounds the loop.

## Exercise 4

`int_widths.py`, on this Windows machine, with the course environment (NumPy 2.4.6, Python 3.14) and with
ToOp's (NumPy 2.3.5, Python 3.11), identically:

| Question | Answer |
|---|---|
| `np.dtype(int)` | `int64` |
| `np.dtype("long")` | `int32` (the C `long` is 32-bit on Windows) |
| `np.dtype(np.intp)` | `int64` |
| `np.random.randint(2**32)` / `default_rng().integers(2**32)` | works (3 653 403 231) |
| `jnp.full(3, 7, dtype=int).dtype` with `jax_enable_x64` | `int64`, and `int_max()` is $2^{63}-1$ |
| the same without x64 (fresh process, `JAX_ENABLE_X64=0`) | `int32`, and `int_max()` is $2^{31}-1$ |
| the int64 maximum stored into an `int32` array | `-1` |

Neither failure of the `int_dtype()` docstring reproduces: `dtype=int` is already 64-bit, so a sentinel of
$2^{63}-1$ fits, and an `int` array meets an `int64` carry in `lax.scan` with equal types. The version
matters because the docstring describes NumPy 1.x: there, `np.dtype(int)` *was* the C `long`, hence `int32`
on Windows, and the two failures were real (`int_max()` truncated to `-1`, which Exercise 3 of
[Chapter 31](../ch31-jax-gpu-fundamentals/index.md) shows reading the last element instead of being dropped,
and an `int32` array against an `int64` carry). NumPy 2.0 changed the default integer on Windows to 64-bit
(`np.dtype(int)` is now `intp` everywhere) while leaving `'long'` on the C type, which is the one row of the
table that still differs between platforms. ToOp pins NumPy 2.x, so the Windows bug is closed by the
dependency; `int_dtype()` is still the right habit, because `jnp.int_` follows `jax_enable_x64` and keeps the
array dtype and `int_max()` consistent with each other by construction, on every platform and in both
precision modes.

## Exercise 5

`pmap_donate.py` sets `XLA_FLAGS=--xla_force_host_platform_device_count=4` before `import jax` (the flag is
read when the backend starts, so inside a notebook that already imported JAX the experiment runs in a
subprocess, `run_in_subprocess()`). `jax.devices()` then lists four CPU devices (`cpu:0`–`cpu:3`;
`TFRT_CPU_0`–`3` under ToOp's JAX 0.5.3). `jax.pmap(lambda x: 2 * x, donate_argnums=(0,))` over a `(4, 3)`
input gives the right output (sum 132), and the donated input is still readable afterwards: its sum is 66 and
`x.is_deleted()` is `False`, with no warning, in both JAX 0.11.2 and 0.5.3. On CPU the donation of `pmap`
arguments is accepted and ignored. `jax.jit(..., donate_argnums=(0,))` on the same CPU *does* delete its input
(`is_deleted()` is `True`, and reading it raises `RuntimeError: Array has been deleted`). *GPU: not run here*;
there the `pmap` input is deleted as well and `float(x.sum())` raises that `RuntimeError`, which is the
behaviour the exercise expects you to see. ToOp donates the result storage (`donate_argnums=(3,)` in the
`jax.pmap` of `iterate_symmetric_sequential` in `topology_looper.py`, and `donate_argnums=(0,)` on the epoch
in `optimizer/dc/worker/optimizer.py`) because that buffer is the largest array the call touches, one row
per topology of sparse N-0 and N-1 results, and the program only ever *overwrites* it slice by slice inside the
`fori_loop` (`dynamic_update_slice_in_dim`). Without donation XLA must keep the input storage intact and
allocate a second buffer of the same size for the output, doubling the peak memory of exactly the array that
sets the batch-size limit; with donation the output reuses the input's device memory. The caller never needs
the zero-filled storage again, so nothing is lost; the price is the rule this exercise demonstrates: a donated
array must not be read afterwards, and on a device that honours donation the error is immediate.

## Exercise 6

*Not run here*: the end-to-end DC stage needs the importer stage (a grid file through
`run_pipeline`/`CgmesImporterParameters`), a static-information file and four optimizer runs of a fixed
number of iterations; that is the exercise in the container. What **was** run is the cost the question is
really about, isolated at the solver level: `split_slots_toop.py` calls `run_solver` on 64 unsplit
topologies whose `action` array has 1, 2, 4, 8 or 16 columns, every slot holding the padding sentinel, so no
substation is ever split and only the *number of slots* changes (this is what `max_num_splits` becomes
inside the optimizer; `default_topology` itself uses one column). CPU, five steady-state repeats:

| Slots | case57 (82 branches, 84 N-1 cases) µs / topology | vs 1 slot | case300 (411 branches, 385 N-1 cases) µs / topology | vs 1 slot |
|---|---|---|---|---|
| 1 | 188 | 1.00 | 1 997 | 1.00 |
| 2 | 198 | 1.05 | 2 261 | 1.13 |
| 4 | 188 | 1.00 | 2 342 | 1.17 |
| 8 | 223 | 1.18 | 2 521 | 1.26 |
| 16 | 251 | 1.33 | 2 941 | 1.47 |

Compile time is 0.8–1.0 s for every row. So yes: the extra slots are paid for even when every slot is empty,
which is the README's statement ("the BSDF module will perform a BSDF computation for every substation
regardless if only a few substations are actually split"), but the bill is smaller than "proportional to
`max_num_splits`" suggests for the *whole* iteration: going from 1 to 16 slots costs +33 % on case57 and
+47 % on case300, because the per-slot BSDF update (a rank-one PTDF update per slot and topology) is cheap
next to the contingency analysis, whose size (N-1 cases × branches) does not depend on the slots. Expect the
e2e plot of seconds per iteration against `max_num_splits` to be a line with a positive slope and a large
intercept: linear in the slots, dominated by the N-1 part, with the mutation and selection stages adding a
slot-independent constant. The practical rule from `example3_e2e_pipeline.ipynb` stands: set
`max_num_splits` to the number of simultaneous splits an operator would accept (2 in the notebook), not to
the number of relevant substations, and spend the saved time on a larger `batch_size`.

```python title="split_slots_toop.py"
--8<-- "docs/part-5-computing/ch32-jax-in-toop/src/split_slots_toop.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch32-jax-in-toop/src){ target="_blank" } to see or download all the files of this chapter.
