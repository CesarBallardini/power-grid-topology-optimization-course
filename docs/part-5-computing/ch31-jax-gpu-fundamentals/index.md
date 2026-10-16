# Chapter 31. JAX and GPU computing fundamentals

**Weeks:** 2 · **Semester:** 3 · **Prerequisites:** [Chapter 30](../ch30-scientific-python/index.md)

!!! info "Why it matters for ToOp"

    The solver and the optimizer are JAX programs, and every JAX idea in this chapter appears in them:

    - `jax.jit` with static and traced arguments (`dc_solver/jax/topology_looper.py`).
    - `jax.vmap` to batch over topologies, injections and outages (`dc_solver/jax/compute_batch.py`).
    - `lax.scan`, `lax.cond`, `lax.fori_loop` and `jnp.where` instead of Python control flow, and fixed shapes with
      padding sentinels so variable-size topologies fit one compiled program (`dc_solver/jax/batching.py`).
    - Equinox pytrees, jaxtyping shape annotations on every array, explicit PRNG keys in the genetic operators
      (`optimizer/dc/genetic_functions/`), forced `float64`, and `pmap` for multiple GPUs.

    This chapter teaches these tools on small, self-contained programs. How ToOp uses them, and the traps specific
    to its data structures, are the subject of [Chapter 32](../ch32-jax-in-toop/index.md).

## Topics

- Functional programming: pure functions, immutability, functional array updates (`x.at[i].set(v)`).
- Tracing and compilation with XLA: abstract values (shape and dtype), jaxprs, the compilation cache; static vs
  traced arguments (`static_argnums`, `static_argnames`); what triggers recompilation (new shapes or dtypes, new
  static values, a new function object such as a fresh `lambda`); `jax.no_tracing()` as a guard in tests.
- Automatic vectorization with `vmap` (`in_axes`, `out_axes`); writing the single-case function first.
- Control flow under `jit`: `jnp.where`, `lax.cond`, `lax.scan`, `lax.while_loop`, `lax.fori_loop` with a traced
  bound; fixed shapes, padding and out-of-bounds sentinels read with `x.at[idx].get(mode="fill", fill_value=...)`;
  why Python `if` on a traced value raises `ConcretizationTypeError` (in recent JAX, its subclass
  `TracerBoolConversionError`).
- Pseudo-random numbers: explicit keys, `jax.random.split`, reproducibility.
- Pytrees; Equinox modules (`eqx.Module`, `eqx.field(static=True)`); jaxtyping shape annotations and runtime
  checking with beartype.
- Numerics: float32 by default vs `jax_enable_x64` (revisits [Chapter 4](../../part-1-math/ch04-numerical-linear-algebra/index.md)
  with a GPU cost attached); integer widths.
- GPU concepts: data parallelism, host–device transfer, asynchronous dispatch and how to time it, memory
  preallocation, batch sizing and out-of-memory errors, float32 vs float64 speed on consumer cards; `pmap` over
  several devices (overview).

## Learning outcomes

- Predict, for a given sequence of calls, which calls of a jitted function compile and which hit the cache, and
  confirm the prediction with `jax.no_tracing()`.
- Rewrite a NumPy routine that uses Python loops and branches as a jitted, vmapped JAX function with fixed shapes.
- Measure compile time and steady-state run time separately on CPU and GPU, and explain when a GPU pays off.
- Annotate arrays with jaxtyping and catch a shape or dtype error at runtime.

## Resources

- JAX documentation (<https://docs.jax.dev/>):
  - "Key concepts" <https://docs.jax.dev/en/latest/key-concepts.html> and "Quickstart: How to think in JAX"
    <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>.
  - "Just-in-time compilation" <https://docs.jax.dev/en/latest/jit-compilation.html>; `jax.no_tracing`
    <https://docs.jax.dev/en/latest/_autosummary/jax.no_tracing.html>.
  - "Automatic vectorization" <https://docs.jax.dev/en/latest/automatic-vectorization.html>; "Pytrees"
    <https://docs.jax.dev/en/latest/pytrees.html>.
  - "Control flow and logical operators with JIT" <https://docs.jax.dev/en/latest/control-flow.html>.
  - "Pseudorandom numbers" <https://docs.jax.dev/en/latest/random-numbers.html>.
  - "🔪 JAX – The Sharp Bits 🔪" (out-of-bounds indexing, float64)
    <https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>.
  - "GPU memory allocation" <https://docs.jax.dev/en/latest/gpu_memory_allocation.html> and "GPU performance tips"
    <https://docs.jax.dev/en/latest/gpu_performance_tips.html>.
- Courses and videos:
  - UvA Deep Learning course (Lippe, University of Amsterdam), *Tutorial 2 (JAX): Introduction to JAX+Flax*
    (notebook, no videos)
    <https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html>.
  - El Hajj (American University of Beirut), *GPU Computing* lectures built on Kirk, Hwu & El Hajj's
    *Programming Massively Parallel Processors* (CUDA C, Spring 2021)
    <https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4>: Lectures 02 "Data Parallel
    Programming", 04 "GPU Architecture", 05 "Memory and Tiling", 16 "Sparse Matrix Computation (COO and CSR)" and
    21 "Pinned Memory and Streams" (host–device transfer). JAX hides the kernels, not the cost model.
  - CMU 10-414/714 *Deep Learning Systems* (Chen), Lecture 12 "GPU Acceleration"
    <https://www.youtube.com/watch?v=jYCxVirq4d0>: the GPU execution model and memory hierarchy, with vector and
    matrix examples. Course page <https://dlsyscourse.org/>.
- Equinox <https://docs.kidger.site/equinox/> and jaxtyping <https://docs.kidger.site/jaxtyping/> ("Runtime type
  checking" <https://docs.kidger.site/jaxtyping/api/runtime-type-checking/>).
- Blondel & Roulet, *The Elements of Differentiable Programming*, arXiv:2403.14606 (free book; background).
- GPU and HPC: NVIDIA, "An Even Easier Introduction to CUDA"
  <https://developer.nvidia.com/blog/even-easier-introduction-cuda/>; Eijkhout, *The Art of HPC* (free)
  <https://theartofhpc.com/>.
- Textbooks: Kirk & Hwu, *Programming Massively Parallel Processors* (2 syllabi; 4th ed. 2022 current), early
  chapters on data parallelism and sparse matrix computation. 2nd ed. 2013 **BORROW**
  [`programmingmassi0000kirk`](https://archive.org/details/programmingmassi0000kirk). Pacheco, *An Introduction
  to Parallel Programming* (2011) **BORROW** [`introductiontopa0000pach`](https://archive.org/details/introductiontopa0000pach).
- ToOp: the JAX rules in `.github/copilot-instructions.md` (jaxtyping shape strings, `eqx.Module` pytrees, static
  vs dynamic information, `jax.debug.print`) and how the code enforces some of them (`F722` ignored in `ruff.toml`,
  `ENABLE_BEARTYPE` in `[tool.pytest_env]` of `pyproject.toml`), as an example of rules a production JAX code base
  enforces.

## Worked example

N-1 screening with line outage distribution factors
([Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md)): for the outage of branch $b$ from bus $f_b$
to bus $t_b$,

$$
\mathrm{LODF}_{a,b} = \frac{\mathrm{PTDF}_{a,f_b}-\mathrm{PTDF}_{a,t_b}}{1-\left(\mathrm{PTDF}_{b,f_b}-\mathrm{PTDF}_{b,t_b}\right)},
\qquad \mathrm{LODF}_{b,b} = -1,
\qquad f^{(b)} = f^{0} + \mathrm{LODF}_{\cdot,b}\, f^{0}_{b}.
$$

Write the function for **one** outage, with no Python branching on array values, then let `vmap` add the batch
axis and `jit` compile the whole batch:

```python title="n1_screen.py"
--8<-- "docs/part-5-computing/ch31-jax-gpu-fundamentals/src/n1_screen.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch31-jax-gpu-fundamentals/src){ target="_blank" } to see or download all the files of this chapter.

Three habits to notice. The bridge case returns a flag instead of raising, because a compiled batch cannot stop
halfway. The outage index is a traced array, so a new outage set of the same length reuses the compiled program;
passing it as a Python tuple in `static_argnums` would compile once per distinct tuple. And GPU timings must wait
for the result (`jax.block_until_ready` or converting to NumPy) and must exclude the first call, which includes
compilation.

## Lab

!!! example "Lab: port N-1 screening from NumPy to JAX"

    **Tools:** Jupyter + NumPy + JAX + matplotlib, in the ToOp container
    ([Chapter 0](../../part-0-orientation/ch00-orientation-and-setup/index.md)), which already has JAX, pandapower,
    jaxtyping and beartype; the `toop-gpu` service if an NVIDIA GPU is available.

    1. **Port.** Take your NumPy PTDF/LODF code from the
       [Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md) lab. Write `n1_flows` for a single
       outage as in the worked example, `vmap` it over all branches, and `jit` it. Check that the result matches
       your NumPy screening to `1e-9` on IEEE 14 and 118.
    2. **Second batch axis.** On IEEE 118, `vmap` again over a batch of 100 random injection vectors (generated
       with `jax.random.split` keys), so one call returns an array of shape `(100, n_branch, n_branch)`.
    3. **Timing.** Build PTDFs for pandapower's `case118`, `case300`, `case1354pegase` and `case2869pegase`.
       For each, time the single-injection screening of step 1 (the $n_{branch}^2$ output of `case2869pegase`
       already takes about 170 MB in float64): the first call and the median of the next ten calls, on CPU
       (`JAX_PLATFORMS=cpu`) and, if available, on GPU, in float64 and in float32. Plot time against number of
       branches on log-log axes. Where does the GPU start to win, if at all? How large is the float64 penalty on
       your card?
    4. **Accidental recompilation.** Make the outage set a Python tuple and mark it static
       (`static_argnums`); call the function with ten different outage sets of the same length and count
       compilations (for example with a Python counter inside the function, which only runs while tracing). Then
       pass a fresh `lambda` post-processing function on every call. Fix both, and prove the fix with
       `jax.no_tracing()`.

## Exercises

1. For `g = jax.jit(f, static_argnums=1)`, predict how many times `f` is traced for the calls
   `g(x3, 2)`, `g(y3, 2)`, `g(x4, 2)`, `g(x3, 3)`, `g(x3.astype(jnp.float32), 2)`, where `x3`, `y3` are float64
   arrays of length 3 and `x4` of length 4. Then check your answer in a notebook. *(Jupyter: JAX)*
2. *(Jupyter: JAX)* Write `clip_negative(x)` with `if x.sum() > 0:` and call it under `jit`. Read the error, then
   fix it twice, with `jnp.where` and with `lax.cond`. Which version evaluates both branches, and when does that
   matter?
3. *(Jupyter: JAX + NumPy)* Pad a list of variable-length index arrays to a fixed length with a sentinel equal to
   the array length, and gather with `x.at[idx].get(mode="fill", fill_value=0.0)`. What happens if the sentinel is
   `-1` instead? Relate the answer to why production code chooses a large out-of-bounds sentinel.
4. *(Jupyter: JAX + matplotlib)* Compute the LODF matrix of IEEE 118 in float32 and float64, and plot the absolute
   difference against $|1-(\mathrm{PTDF}_{b,f_b}-\mathrm{PTDF}_{b,t_b})|$. Which outages lose the most accuracy in
   float32, and what does that say about near-bridges ([Chapter 4](../../part-1-math/ch04-numerical-linear-algebra/index.md))?
5. *(Octave)* Implement N-1 screening of IEEE 118 twice in Octave: a `for` loop over outages, and one vectorized
   matrix expression. Compare run times. Which JAX transformation plays the role of your vectorized rewrite, and
   what does `jit` add on top?
6. *(Jupyter: JAX + jaxtyping)* Annotate `n1_flows` with `Float[Array, " n_branch n_bus"]`-style shapes, decorate
   it with `@jaxtyped(typechecker=beartype)`, and call it with a PTDF whose number of columns does not match the
   injection vector. What error do you get, and at which moment (tracing or execution)?
