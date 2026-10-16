# Chapter 31 – Solutions

Solutions of the Lab and the Exercises of [Chapter 31](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`n1_screen.py`](src/n1_screen.py) (the worked example),
[`grid_data.py`](src/grid_data.py), [`n1_batch.py`](src/n1_batch.py), [`timing_sweep.py`](src/timing_sweep.py),
[`compile_count.py`](src/compile_count.py), [`control_flow.py`](src/control_flow.py),
[`lodf_precision.py`](src/lodf_precision.py), [`shape_check.py`](src/shape_check.py) and
[`n1_loop_vs_vec.m`](src/n1_loop_vs_vec.m), each with a test next to it. Every script prints the numbers quoted
below when run from the `src` folder. All JAX numbers are **CPU** numbers (JAX 0.11 on the course environment,
Python 3.14, Windows; the laptop has no NVIDIA GPU), with `jax_enable_x64` on; the jaxtyping exercise ran with
ToOp's environment (jaxtyping 0.2.38, beartype 0.22, JAX 0.5.3, ToOp commit `ba1874e`), which is the one that
has those libraries. GPU behaviour is described, not measured.

## Lab

1. **Port.** `n1_flows` in `n1_screen.py` is the single-outage function: the LODF column
   $(\mathrm{PTDF}_{\cdot,f_b}-\mathrm{PTDF}_{\cdot,t_b})/(1-(\mathrm{PTDF}_{b,f_b}-\mathrm{PTDF}_{b,t_b}))$, the
   diagonal entry forced to $-1$ with `jnp.where`, the post-contingency flows $f^0 + \mathrm{LODF}_{\cdot,b} f^0_b$,
   and a boolean flag `|denom| > 1e-11` in place of the `ValueError` the NumPy loop raises for a bridge.
   `screen = jax.jit(jax.vmap(n1_flows, in_axes=(None, None, None, None, 0)))` adds the outage axis.
   `grid_data.branch_data` pulls `from_bus`, `to_bus`, `x` and the injections out of pandapower through its
   MATPOWER converter (bus numbers consecutive from zero, lines and transformers both branches), and
   `grid_data.numpy_ptdf`/`numpy_screen` are the Chapter 17 loop. On IEEE 14 (20 branches, 1 bridge) and
   IEEE 118 (186 branches, 9 bridges) the JAX flows match the NumPy flows to `1e-9` on every non-bridge outage,
   the two PTDFs match to `1e-9`, and the flags match the bridges the loop skipped (`test_grid_data.py`).
2. **Second batch axis.** `n1_batch.screen_injections = jax.jit(jax.vmap(screen, in_axes=(None, None, None, 0,
   None)))`: a `vmap` of the already jitted, already vmapped screening, with the batch on the injection
   argument only. `random_injections` splits one key into 100 subkeys and perturbs the base injection by
   10 % Gaussian noise. One call on IEEE 118 returns `flows` of shape `(100, 186, 186)` in float64 and `ok` of
   shape `(100, 186)`; the 9 bridges are flagged in every one of the 100 injections (900 `False` flags in all,
   since a bridge is a property of the topology, not of the injection); the largest post-contingency flow of a
   batch member ranges from 362.2 to 624.0 MW; the same key gives the same batch (`True`).
3. **Timing.** `timing_sweep.time_screen` clears the compilation cache, times the first call (compilation
   included) and the median of the next ten, every call wrapped in `jax.block_until_ready`. The PTDF is built
   in float64 and cast, so the float32 rows time the screening itself in float32. CPU, one run:

    | Grid | Branches | Output | float64 first [s] | float64 steady | float32 first [s] | float32 steady |
    |---|---|---|---|---|---|---|
    | case118 | 186 | 0.3 MB | 0.119 | 0.1 ms | 0.090 | 0.1 ms |
    | case300 | 411 | 1.4 MB | 0.108 | 0.7 ms | 0.099 | 0.4 ms |
    | case1354pegase | 1 991 | 31.7 MB | 0.696 | 26.9 ms | 0.104 | 8.9 ms |
    | case2869pegase | 4 582 | 168 MB | 1.126 | 156 ms | 0.194 | 104 ms |

    On log-log axes the steady-state line has slope close to 2 between 411 and 4 582 branches (the output is
    $n_{branch}^2$ numbers, and the work is a rank-one update per outage), and the first call is almost flat at
    0.1 s for the small grids: that is compilation, which does not depend on the data, plus the one-time
    backend start-up in the first row. The float64 penalty on this CPU is 1.5–3× in the steady state
    (26.9 vs 8.9 ms, 156 vs 104 ms), mostly memory traffic: the float64 output of `case2869pegase` is 168 MB,
    twice the float32 one. *GPU: not run here.* Expect the first call to be no faster (compilation is the same
    work, and the arrays must be copied to the device), the steady state of the two small grids to be *slower*
    than CPU (a few tens of microseconds of launch latency per call dominate a 0.1 ms job), and the GPU to win
    from roughly a thousand branches up, by one to two orders of magnitude on the Pegase cases in float32. In
    float64 a consumer GeForce card has about 1/32 of its float32 arithmetic throughput, so the float64 penalty
    is large where the kernel is compute-bound and about 2× where it is bandwidth-bound; a data-centre card
    (A100, H100) has a 2× penalty throughout. Measure it with the same two numbers per grid and put the four
    lines (CPU/GPU × float32/float64) on the same plot.

    ```python title="timing_sweep.py"
    --8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/timing_sweep.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

4. **Accidental recompilation.** `compile_count.counting` wraps a function so that a Python counter runs on every
   Python-level call; under `jit` the Python body only runs while tracing, so the counter counts
   compilations. With the outage set as a Python tuple marked static (`screen_static`), every distinct tuple is
   a new cache key: on the four-bus grid, 4 calls with 4 distinct pairs give 4 traces, and in general ten
   different outage sets give ten compilations, even though all have the same length (the test checks that the
   count equals the number of *distinct* tuples, and that a second round of the same tuples adds none). With a
   fresh `lambda` passed as the static `post` argument (`screen_then`), 10 calls give 10 traces: every `lambda`
   expression evaluates to a new function object, and function objects hash by identity. The fixes: pass the
   outages as a `jnp` array, so only its shape and dtype matter (`screen` in the worked example), and define the
   post-processing function once at module level (`worst_flow`): 10 calls, 1 trace. The proof: after one
   warm-up call per shape, the fixed versions run inside `with jax.no_tracing():` without error, while a call
   with a new outage *length* inside that block raises `RuntimeError` (`test_compile_count.py`). Build the index arrays *before* entering the block: creating an array is itself a dispatch that `no_tracing` flags the first time it sees that shape and dtype.

    ```python title="compile_count.py (the static tuple)"
    --8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/compile_count.py:static"
    ```

    ```python title="compile_count.py (the fresh lambda and the fix)"
    --8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/compile_count.py:lambda"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Four traces. The cache key of a jitted function is the abstract value (shape and dtype) of every traced
argument plus the value of every static one:

| Call | Key | Traced? | Traces so far |
|---|---|---|---|
| `g(x3, 2)` | `f64[3]`, 2 | yes, first call | 1 |
| `g(y3, 2)` | `f64[3]`, 2 | no: same shape, dtype and static value; the *values* of `x3` and `y3` do not matter | 1 |
| `g(x4, 2)` | `f64[4]`, 2 | yes: new shape | 2 |
| `g(x3, 3)` | `f64[3]`, 3 | yes: new static value | 3 |
| `g(x3.astype(jnp.float32), 2)` | `f32[3]`, 2 | yes: new dtype | 4 |

`compile_count.exercise_1_traces()` counts them with the wrapper of Lab 4 and returns `[1, 1, 2, 3, 4]`. With
`jax.no_tracing()` around the second call nothing happens; around any of the other four it raises.

## Exercise 2

Eagerly, `clip_negative_python` works (`[0, 2, 0, 5]` for `x = [-1, 2, -3, 5]`). Under `jit` the `if` asks
`bool()` of a tracer and raises `TracerBoolConversionError` (a `ConcretizationTypeError`): while tracing, JAX
knows the shape and dtype of `x.sum() > 0` but not its value, so it cannot pick a branch. Both fixes give the
same result under `jit`:

```python title="control_flow.py (the three versions)"
--8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/control_flow.py:clip"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

`jnp.where` evaluates **both** branches on every call and selects per element; `lax.cond` compiles both but
executes only the taken one (its predicate must be a scalar). It matters when a branch is expensive (you pay for
both with `where`), when the untaken branch produces `inf` or `NaN` that leak through gradients, and when a
branch has side effects (`jax.debug.print`). The gradient shows it: for `safe_sqrt_where(x) = where(x > 0,
sqrt(x), 0)`, `jax.grad` at $x=-1$ is `nan`, because the derivative of `sqrt` is evaluated at $-1$ and
$0 \times \text{NaN}$ is NaN; with `lax.cond` it is `0.0`. Under `vmap`, however, `cond` is lowered to a
`select_n` over both branches (`jax.make_jaxpr(jax.vmap(clip_negative_cond))` contains `select_n`), because
different batch members may take different branches: inside a vmapped, jitted solver like ToOp's, `cond` buys
nothing over `where`, and the code uses `jnp.where` and masks throughout.

## Exercise 3

`pad_indices([[0, 1, 2], [3], [1, 3]], sentinel=4)` gives `[[0, 1, 2], [3, 4, 4], [1, 3, 4]]` and
`x.at[idx].get(mode="fill", fill_value=0.0)` on `x = [10, 20, 30, 40]` returns `[[10, 20, 30], [40, 0, 0],
[20, 40, 0]]`: index 4 is out of bounds, so the slot reads the fill value and the row sums `[60, 40, 60]` are the
sums of the real members. With `sentinel=-1` the gather returns `[[10, 20, 30], [40, 40, 40], [20, 40, 40]]`
and the row sums become `[60, 120, 100]`: JAX normalizes negative indices like NumPy does, so `-1` is the *last
element*, in bounds, and every padding slot silently adds `x[-1]`. Nothing fails; the numbers are just wrong.
That is why production code pads with a large positive sentinel (ToOp's `int_max()`,
[Chapter 32](../ch32-jax-in-toop/index.md)): it is out of bounds for every array the code will ever index, so
`mode="fill"` drops it, and if it ever shows up in a result it is an obviously impossible index rather than a
plausible one. The same reasoning says why an `int64` sentinel stored in an `int32` array is dangerous: it wraps
to `-1`, which is this exercise's bad case.

```python title="control_flow.py (padding and gathering)"
--8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/control_flow.py:pad"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`lodf_precision.lodf_matrix` computes the whole LODF matrix at once, in the dtype of the PTDF it is given;
the study builds the PTDF of IEEE 118 in float64 and casts it to float32 for the second run, so the difference
measures what the LODF formula itself does to float32 round-off (a float32 PTDF solve would add its own error
of order $10^{-6}$ relative on top). IEEE 118 has 9 bridges, with $|1-(\mathrm{PTDF}_{b,f_b}-\mathrm{PTDF}_{b,t_b})| < 10^{-11}$;
their columns are infinite in both precisions and are excluded. Over the 177 other outages the largest
float32 error of a column has median $1.1 \times 10^{-7}$ and maximum $5.5 \times 10^{-7}$, and it grows as the
denominator shrinks: median $2.5 \times 10^{-7}$ for $|\text{denom}| < 0.2$ against $5.7 \times 10^{-8}$ for
$|\text{denom}| \ge 0.5$, with a correlation of $-0.79$ between $\log_{10}$ error and $\log_{10} |\text{denom}|$.
The five smallest denominators among non-bridges are the outages of branches 114–115 (0.053), 70–71 (0.059),
108–109 (0.069), 4–5 (0.072) and 65–68 (0.080), with errors between $1.6 \times 10^{-7}$ and
$5.1 \times 10^{-7}$. On the plot (error against $|\text{denom}|$, both axes logarithmic) the cloud slopes
down to the right: these are the near-bridges. A branch whose outage leaves only a weak parallel path has
$\mathrm{PTDF}_{b,f_b}-\mathrm{PTDF}_{b,t_b}$ close to 1, so the division by $1-(\cdot)$ amplifies the
round-off of the numerator by $1/|\text{denom}|$: a condition number, in the sense of
[Chapter 4](../../part-1-math/ch04-numerical-linear-algebra/index.md), that is set by the topology, not by the
solver. In float64 the same amplification acts on $10^{-16}$ and is harmless; in float32 it eats a quarter of
the seven significant digits for the worst outages here, and all of them on a grid with a denominator of
$10^{-4}$. This is why ToOp forces `float64` even on GPUs that pay for it.

```python title="lodf_precision.py"
--8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/lodf_precision.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

Both versions start from MATPOWER's `makePTDF` and the base-case flows $f^0 = H P$. The loop takes one outage
at a time: one PTDF column difference, one scalar denominator, one rank-one update. The vectorized version
forms all column differences at once, `col = H(:, f) - H(:, t)` (an $n_l \times n_l$ matrix whose column $k$
belongs to outage $k$), divides every column by its own denominator with broadcasting, sets the diagonal to
$-1$, blanks the bridge columns, and builds every post-contingency flow vector in one expression
`F = f0' + L' .* f0`. On IEEE 118 (Octave 11.3, MATPOWER 8.1, CPU) the two give identical matrices (largest
difference exactly 0, the same 9 bridge rows) and take 3.2–3.7 ms for the loop against 0.4–0.5 ms vectorized,
a factor of about 8. The vectorized rewrite is what `jax.vmap` does for you: it takes the one-outage
function and adds the batch axis mechanically, with no hand-written broadcasting. `jit` adds what no array
language can: it compiles the whole batched program into one fused XLA executable, so the intermediate
$n_l \times n_l$ matrices (`col`, `L`, `L'`) need not all be materialized, Python and interpreter overhead
disappear from the steady state, and the same program runs unchanged on a GPU. One finding on the way:
`makeLODF(mpc, H, true)` (bridge masking) of MATPOWER 8.1 stops with `br(197): out of bound 186` inside
`find_bridges` on `case118`, so the file masks the bridges itself.

```octave title="n1_loop_vs_vec.m"
--8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/n1_loop_vs_vec.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 6

```python title="shape_check.py (the annotated function)"
--8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/shape_check.py:annotated"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

A valid call gives the worked example's first row, `[0, 100, -25, 25, 75]` with the flag `True`. Called with a
PTDF of 4 columns and an injection vector of 5 entries, the function raises `jaxtyping.TypeCheckError` (a
`TypeError` subclass) whose message names the mismatch: "The problem arose whilst typechecking parameter 'p'.
Actual value: `f64[5]`. Expected type: `Float[Array, 'n_bus']`", followed by the shapes of every argument and
the current bindings `n_branch=5`, `n_bus=4` (set by `ptdf`, which was checked first). Eagerly it happens at
call time, before any arithmetic. Under `jit` it happens **while tracing**: the tracers
that stand for the arguments carry concrete shapes and dtypes, which is all jaxtyping checks, so the error
appears on the first call for a given set of shapes, with the same message, and never at execution. The flip
side is that a *cached* call does not run the Python body at all, so it is not checked again, which is fine,
since its shapes are by definition the ones that passed. This is also why ToOp can leave `ENABLE_BEARTYPE` on
in its test configuration at no run-time cost: the checks run once per compilation, not once per batch. The
course environment does not ship jaxtyping and beartype; `test_shape_check.py` skips there and passes with
ToOp's interpreter.
