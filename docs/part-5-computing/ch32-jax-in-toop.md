# Chapter 32. JAX in ToOp: internals and GPU tuning

**Weeks:** 1.5 · **Semester:** 4 · **Prerequisites:** [Chapter 18](../part-3-power-systems/ch18-sensitivity-factors-2.md), [Chapter 31](ch31-jax-gpu-fundamentals.md)

!!! info "Why it matters for ToOp"

    ToOp's speed depends on compiling the solver once and feeding it millions of topologies; its correctness
    depends on the compiled program seeing the data you think it sees. Both are governed by a few design choices:

    - **What is static.** `StaticInformation` (`dc_solver/jax/types.py`) holds a traced
      `dynamic_information: DynamicInformation` and a static field
      `solver_config: Static[SolverConfig] = eqx.field(static=True)`. The jitted core
      `iterate_symmetric_sequential` (`dc_solver/jax/topology_looper.py`) is declared with
      `static_argnames=("solver_config", "aggregate_output_fn")`: those two arguments select the compiled
      program, while `DynamicInformation` (PTDF, injections, limits, action set) is traced. The recompilation
      trigger is `SolverConfig` plus the aggregation function, not `StaticInformation` as a whole (ToOp's
      `packages/dc_solver_pkg/README.md` states the coarser version).
    - **How static values are compared.** `SolverConfig.__hash__` and `StaticInformation.__hash__` both return
      `id(self)`, and `__eq__` is `self is other`. The compile cache is keyed on *object identity*, not on values.
    - **Fixed shapes.** Padding with the `int_max()` sentinel, read through `.at[idx].get(mode="fill")`
      (`dc_solver/jax/batching.py`, `dc_solver/jax/lodf.py`), and loops with traced bounds
      (`lax.fori_loop` in `optimizer/dc/genetic_functions/mutation/mutate.py`).
    - **Memory and devices.** `batch_size`, `batch_size_bsdf` and `batch_size_injection` trade memory for
      throughput; `jax.pmap` with buffer donation spreads batches over GPUs (`dc_solver/jax/topology_looper.py`,
      `optimizer/dc/worker/optimizer.py`).

## Topics

- The static/dynamic split in ToOp: what `SolverConfig` holds (`batch_size_bsdf`, `batch_size_injection`, `slack`,
  `n_stat`, `number_most_affected`, `contingency_ids`, `enable_bb_outages`, `distributed`, ...) and what
  `DynamicInformation` holds (`ptdf`, `nodal_injections`, `branch_limits`, `action_set`, `unsplit_flow`,
  `branches_to_fail`, ...); functional updates with `jax_dataclasses.replace`.
- Identity hashing and its consequences: `replace(solver_config)` with unchanged values still recompiles; mutating
  a field in place (`solver_config.number_most_affected = 20`) reuses the old compiled program; a fresh `lambda`
  as `aggregate_output_fn` recompiles every call (you measure the first, cached and fresh-lambda call times on
  IEEE 57 in the Lab); every static field (for example `number_most_affected`) recompiles when changed.
- Hidden constants: `run_solver` builds a `DefaultAggregateOutputFn` whose hash is `hash(solver_config)` and which
  stores `max_mw_flow` and `branches_to_fail` as Python attributes (`dc_solver/jax/topology_looper.py`). Those
  arrays are baked into the compiled program as constants (see the Lab).
- The optimizer loop: `run_single_device_epoch` is jitted with `iterations_per_epoch` and `update_fn` static, and
  runs `lax.fori_loop` over `run_single_iteration`, which donates `jax_data` (`optimizer/dc/worker/optimizer.py`).
  `packages/topology_optimizer_pkg/tests/test_jax_compilation.py` uses `jax.no_tracing()` to prove that later
  epochs do not retrace and that changing `iterations_per_epoch` or the batch size does.
- Data-dependent loop counts without recompiling: the number of substation mutations is a Poisson draw clipped to
  `[1, max_num_splits]` and used as the traced upper bound of `jax.lax.fori_loop` (`mutate.py`); `fori_loop` over
  batches in `topology_looper.py`.
- Sentinels and integer widths: `int_max()` (`dc_solver/jax/types.py`) as an out-of-bounds pad value, dropped by
  `get(mode="fill", fill_value=...)`; why a sentinel that wraps to `-1` corrupts data instead of being ignored;
  the `int_dtype()` helper introduced on the course fork's `feat/add-docker-compose` branch (not in upstream `main`),
  whose docstring attributes Windows-only failures to `dtype=int` following the platform's C `long` (Exercise 4).
- Multiple devices: `lf_config.distributed=True` → `jax.pmap` with `static_broadcasted_argnums=(6, 7)` and
  `donate_argnums=(3,)` (the result storage) in `topology_looper.py`, and `jax.pmap` of the epoch with
  `donate_argnums=(0,)` in `optimizer/dc/worker/optimizer.py`; faking several CPU devices with
  `XLA_FLAGS=--xla_force_host_platform_device_count=N` (`optimizer/benchmark/benchmark_utils.py`,
  `dc_solver/jax/benchmarks/runner.py`).
- Memory and throughput knobs, in the order to cut on out-of-memory: `lf_config.batch_size` ("the largest value
  that fits in your VRAM" in `notebooks/example3_e2e_pipeline.ipynb`; the DC optimizer copies it into
  `batch_size_bsdf` in `optimizer/dc/genetic_functions/initialization.py`), then, for direct solver calls,
  `solver_config.batch_size_bsdf`, then `batch_size_injection` (both named for out-of-memory errors in "Debugging
  Tips" of `.github/copilot-instructions.md`). BSDF work is proportional to `max_num_splits`, and disconnection
  work to `max_num_disconnections`, even when the slots are unused (`packages/dc_solver_pkg/README.md`). The
  README calls the latter "a LODF computation for every slot"; in the batch path `apply_disconnections`
  (`dc_solver/jax/disconnections.py`, called from `dc_solver/jax/compute_batch.py`) applies all slots as one joint
  multi-outage (MODF) update of size `max_num_disconnections`, and a failed solve marks the whole topology
  unsuccessful. The batch-size search notebook
  `dc_solver/jax/parse_batch_size_finder.ipynb` plots mean time over a grid of `(batch_size_bsdf,
  batch_size_injection)` benchmark runs.
- Precision and environment: `jax_enable_x64` forced in `optimizer/dc/main.py` and
  `dc_solver/preprocess/convert_to_jax.py`; `runner.py` sets `JAX_ENABLE_X64`, `CUDA_VISIBLE_DEVICES`,
  `OMP_NUM_THREADS` and `XLA_FLAGS` per benchmark and runs each benchmark in a spawned process, because these
  variables only take effect before JAX initializes; float64 on consumer GeForce cards reported at about 1/32 of
  float32 speed (the GPU banner of the course fork's `docker/device_banner.py`); `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  on small GPUs.

## Learning outcomes

- Given a change to a ToOp input (a `SolverConfig` field, a `DynamicInformation` array, the batch size, the
  aggregation function), predict whether the solver recompiles, reuses the compiled program correctly, or reuses
  it incorrectly, and prove it with timings and `jax.no_tracing()`.
- Measure compile time, steady-state throughput and peak memory of `run_solver` as a function of the batch sizes,
  and choose settings for a given device.
- Explain how padding sentinels, traced loop bounds and buffer donation keep ToOp's shapes fixed and its memory
  bounded.

## Resources

- ToOp sources, read in this order: `dc_solver/jax/types.py` (`SolverConfig`, `DynamicInformation`,
  `StaticInformation`), `dc_solver/jax/topology_looper.py` (`run_solver`, `DefaultAggregateOutputFn`,
  `run_solver_symmetric`, `iterate_symmetric_sequential`), `dc_solver/jax/batching.py`,
  `optimizer/dc/worker/optimizer.py`, `packages/topology_optimizer_pkg/tests/test_jax_compilation.py`.
- ToOp documentation: `packages/dc_solver_pkg/README.md` (static/dynamic information, cost of BSDF and LODF),
  `docs/dc_solver/quickstart.md` (`jax_enable_x64`), and the JAX rules in `.github/copilot-instructions.md`
  ("Static vs Dynamic Info", "JAX Type Annotations", "JAX Debugging").
- JAX documentation: `jax.jit` (`static_argnums`, `static_argnames`, `donate_argnums`)
  <https://docs.jax.dev/en/latest/_autosummary/jax.jit.html>; `jax.pmap`
  <https://docs.jax.dev/en/latest/_autosummary/jax.pmap.html>; `jax.lax.fori_loop`
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html>; "FAQ" (buffer donation)
  <https://docs.jax.dev/en/latest/faq.html>; "GPU memory allocation"
  <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>; "XLA compiler flags"
  <https://docs.jax.dev/en/latest/xla_flags.html>; "GPU performance tips"
  <https://docs.jax.dev/en/latest/gpu_performance_tips.html>.
- Equinox, static fields: <https://docs.kidger.site/equinox/>.
- Paper: "Accelerated DC loadflow solver for topology optimization", arXiv:2501.17529 (the solver paper).

## Lab

!!! example "Lab: compile time, batch size and the recompilation trigger"

    **Tools:** the ToOp container (CPU; the `toop-gpu` service if available), Jupyter + JAX + pandas +
    matplotlib; `notebooks/example1_dc_loadflow_example.ipynb`.

    1. **Setup.** As in `notebooks/example1_dc_loadflow_example.ipynb`, write `case57_data_powsybl` to a folder,
       call `load_grid`, and keep `sc = static_information.solver_config` and
       `di = static_information.dynamic_information`. Build `topologies = default_topology(sc, batch_size=4)`.
       Always wait for results (`jax.block_until_ready`) before stopping a timer.
    2. **Compile vs run.** Time `run_solver(topologies, None, None, di, sc)` three times. Then call
       `run_solver_symmetric` with a pass-through aggregation function defined once, and with a fresh `lambda` on
       every call; tabulate the first-call, cached and fresh-`lambda` times and explain the pattern.
    3. **Find the trigger.** For each change below, first predict "recompiles", "reuses correctly" or "reuses
       incorrectly", then measure the time and inspect `result.n_1_results.pf_n_1_max` (shape and maximum):

        - `replace(sc)` with no field changed;
        - `replace(sc, number_most_affected=20)`, and separately the in-place `sc.number_most_affected = 20`;
        - a new `DynamicInformation` with `nodal_injections` scaled by 1.1 (the flows do not change: find in
          `dc_solver/jax/compute_batch.py` which field the N-0 flows are updated from, and scale that one too);
        - a new `DynamicInformation` whose `branch_limits.max_mw_flow` is halved, passed with the *same* `sc`,
          and then with `replace(sc)`;
        - a batch of 8 topologies instead of 4.

        Wrap the calls you expect to hit the cache in `with jax.no_tracing():` to turn a silent recompile into an
        error. Explain every result from `static_argnames`, `SolverConfig.__hash__` and the attributes of
        `DefaultAggregateOutputFn`. On ToOp commit `ba1874e` (CPU, JAX 0.5.3), halving the limits with the same
        `sc` returned the *old* relative loadings with no recompile, while `replace(sc)` recompiled and doubled
        them: write this up as a bug report with a minimal reproduction.
    4. **Batch size and memory.** On `case300_powsybl` (from `toop_engine_dc_solver.example_grids`), sweep
       `batch_size_bsdf` and `batch_size_injection` over powers of two with a fixed number of topologies. Record
       compile time, topologies per second and peak memory (`nvidia-smi` on GPU with
       `XLA_PYTHON_CLIENT_PREALLOCATE=false`; `docker stats` on CPU, stopping before the container limit). Draw a
       heat map as in `dc_solver/jax/parse_batch_size_finder.ipynb`, find the out-of-memory boundary on GPU, and
       repeat one point with `JAX_ENABLE_X64` off to measure the float64 penalty on your device.

## Exercises

1. Explain why `SolverConfig` hashes by `id(self)` instead of by value. Name one benefit (cost of hashing large
   `HashableArrayWrapper` fields) and two risks, and propose a safer design (for example a frozen dataclass with
   a value hash computed once).
2. *(Jupyter: JAX)* Reproduce the stale-result trap without ToOp: a jitted function with a static argument whose
   class defines `__hash__` as `id(self)` and `__eq__` as identity. Show that mutating a field in place returns
   the old result, and that `dataclasses.replace` recompiles. Then write a `pytest` test that would have caught it.
3. *(Jupyter: JAX + NumPy)* In `mutate.py` the loop bound `n_subs_mutated` is a traced Poisson draw. Write a toy
   version with `lax.fori_loop` and a traced bound, and a second one with a Python `for` over the same draw. Which
   one compiles once, and what does `jax.make_jaxpr` show for each?
4. *(Jupyter: NumPy)* Check on your machine what `np.dtype(int)`, `np.dtype("long")` and `np.random.randint(2**32)`
   do, and what `jnp.full(3, 7, dtype=int).dtype` is with and without `jax_enable_x64`. The `int_dtype()` docstring
   on the course fork's `feat/add-docker-compose` branch attributes two Windows-only failures to platform integer
   widths (a padding sentinel stored as `-1`, a `jax.lax.scan` carry type mismatch); which can you reproduce with
   ToOp's pinned NumPy 2.x, and why does NumPy's version matter?
5. *(Jupyter: JAX)* Set `XLA_FLAGS=--xla_force_host_platform_device_count=4` before importing JAX, `pmap` a
   function over a leading axis of size 4 with `donate_argnums=(0,)`, and try to read the donated input afterwards.
   Why does ToOp donate the result storage in `topology_looper.py`?
6. *(Jupyter: pandas + matplotlib)* Run the DC stage of `notebooks/example3_e2e_pipeline.ipynb`
   (`run_dc_optimization_stage`) for a fixed number of iterations with
   `lf_config.max_num_splits` set to 1, 2, 4 and 8 (keep `split_subs` descriptor cells at most 2, so every value
   is valid) and plot seconds per iteration against `max_num_splits`. Compare with the cost statement in
   `packages/dc_solver_pkg/README.md`: are the extra split slots paid for even when most topologies use one?
