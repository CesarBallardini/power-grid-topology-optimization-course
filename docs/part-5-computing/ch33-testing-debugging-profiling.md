# Chapter 33. Testing, debugging and profiling scientific and JAX code

**Weeks:** 2 · **Semester:** 4 · **Prerequisites:** [Chapter 32](ch32-jax-in-toop.md)

!!! info "Why it matters for ToOp"

    Topology optimization code fails quietly. A wrong sign still produces plausible flows, a compiled batch
    cannot raise halfway, and a mistyped parameter is accepted. ToOp's safety net is worth studying, and so are
    its holes:

    - **Success flags instead of exceptions.** Near-singular cases return `success = False` rather than raising:
      `|denom| > 1e-11` for LODF (`dc_solver/jax/lodf.py`), `|denom| >= 1e-5` for BSDF (`dc_solver/jax/bsdf.py`,
      where an `equinox.error_if` check is left commented out), `|det| > 1e-10` plus a finiteness check for small
      multi-outage systems (`dc_solver/jax/unrolled_linalg.py`).
    - **Reference and oracle tests.** A plain NumPy reference (`packages/dc_solver_pkg/tests/numpy_reference.py`)
      checks the JAX kernels (`packages/dc_solver_pkg/tests/jax/test_lodf.py`, `test_bsdf.py`), and the solver is
      compared with the backends' own load flows
      (`packages/dc_solver_pkg/tests/preprocessing/test_loadflows_match.py`).
    - **Runtime shape checks.** `ENABLE_BEARTYPE="true"` in `[tool.pytest_env]` of ToOp's `pyproject.toml` makes each
      package `__init__.py` install a jaxtyping import hook with beartype.
    - **Gates.** A 90% coverage threshold (`.coveragerc`), compilation guards with `jax.no_tracing()`
      (`packages/topology_optimizer_pkg/tests/test_jax_compilation.py`), and example notebooks executed as tests
      (`notebooks/tests/test_notebooks.py`).
    - **Holes.** Pydantic parameter models that silently drop unknown keywords (they keep Pydantic's default
      `extra="ignore"`, e.g. `interfaces/messages/preprocess/preprocess_commands.py`), a notebook glob that
      skips non-`example*` notebooks, stale compiled results ([Chapter 32](ch32-jax-in-toop.md)), and no
      property-based tests.

## Topics

- **Debugging numerics.** Where NaN and inf come from in sensitivity factors (bridges, islanding, near-singular
  denominators); `jax_debug_nans`, `jax.debug.print`, `jax.debug.breakpoint`, `jax.debug.callback`; success flags
  vs exceptions vs `equinox.error_if`; the reference raises `ValueError` for a bridge while the JAX kernel returns a
  flag. Why ToOp's three thresholds differ: the LODF denominator
  $1-(\mathrm{PTDF}_{b,f_b}-\mathrm{PTDF}_{b,t_b})$ is dimensionless and lies in $[0,1]$ for a meshed grid, the
  BSDF denominator is a sum of susceptances (grid units and size enter), and a determinant scales with the entries
  of the matrix to the power of its size; none of the three is scale-invariant (revisits
  [Chapter 4](../part-1-math/ch04-numerical-linear-algebra.md)). ToOp does not document its choice.
- **Debugging JAX.** `ConcretizationTypeError` and `TracerBoolConversionError`; recompilation traps and stale
  compiled results ([Chapter 32](ch32-jax-in-toop.md)); jaxtyping and beartype shape and dtype errors (the hook
  only works if `ENABLE_BEARTYPE` is set before the package is imported); integer widths on Windows (the
  `int_dtype()` helper on the course fork, [Chapter 32](ch32-jax-in-toop.md)); out-of-memory on GPU (preallocation,
  batch sizes) and in containers (exit code 137, `/dev/shm`).
- **Silent no-ops.** Pydantic models without `extra="forbid"` ignore unknown keyword arguments:
  `PreprocessParameters(enable_bb_outage=True)` is accepted and does nothing, because the preprocessing field is
  `preprocess_bb_outages` (`interfaces/messages/preprocess/preprocess_commands.py`), while `ActionSet` does forbid
  extras (`interfaces/stored_action_set.py`); dead configuration keys (`double_precision`, `output_json`); a GPU
  container that silently runs on CPU ([Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md)).
- **Testing strategies.**
    - Reference implementations and oracle tests against independent solvers. An oracle only helps if it solves the
      same problem: ToOp's `test_loadflows_match.py` compares absolute flows with `rtol=1e-3, atol=1e-3`, which hides
      sign-convention errors, and `packages/dc_solver_pkg/tests/jax/test_busbar_outage.py` runs its pypowsybl oracle
      with `SINGLE_SLACK` while ToOp's example grids store `CGMES_DISTRIBUTED_SLACK` load flow parameters
      (`grid_helpers/powsybl/loadflow_parameters.py`), so check the slack convention of every oracle.
    - Tolerance-based assertions (`np.testing.assert_allclose`, relative vs absolute tolerance) and how to justify
      a tolerance.
    - Property-based testing with Hypothesis on invariants: the LODF diagonal is $-1$ so the outaged branch carries
      no flow; N-1 flows from LODFs equal a re-solve without the branch; a BSDF-updated PTDF equals the PTDF rebuilt
      from the split network; Kirchhoff's current law holds at both busbars of a split.
    - Regression tests and fixtures (`packages/dc_solver_pkg/tests/test_example_grids.py` regenerates tracked grid
      files: check `git status` after a run); compilation guards with `jax.no_tracing()`.
    - Coverage as a gate and what it does not measure; notebooks as tests, and the gap in
      `notebooks/tests/test_notebooks.py`, which globs `example*.ipynb` only, so
      `notebooks/run_ac_contingency_analysis.ipynb` and `notebooks/openrao.ipynb` never run in CI.
- **Profiling.** JAX profiler traces viewed in Perfetto or XProf (TensorBoard profiling), device memory profiling;
  py-spy for Python-level hot spots and hangs (`py-spy record`, `py-spy dump`; ToOp's `docker/README.md` on the
  course fork uses it to diagnose a pytest-xdist deadlock); the Ray dashboard for pandapower contingency analysis,
  which runs through `ray.remote` (`contingency/pandapower/contingency_analysis_pandapower.py`); timing discipline
  (warm-up, `jax.block_until_ready`, repeats, medians).
- **Seeded bugs.** Deliberately planted faults (PSDF sign flip, float32, wrong slack, recompiling lambda) as a way
  to practise the full loop: symptom → localization → failing test → fix.

## Learning outcomes

- Locate the origin of a NaN, a wrong sign or a silent no-op in NumPy or JAX power-flow code, using debug prints,
  JAX debug flags and bisection against a reference.
- Write oracle, tolerance-based and property-based tests for sensitivity-factor code, and justify each tolerance.
- Profile a ToOp run and attribute wall-clock time to compilation, device computation, Python overhead and CPU
  workers.
- Find and fix a planted bug in a ToOp checkout, with a regression test that fails before the fix and passes after.

## Resources

- JAX documentation: "Introduction to debugging" <https://docs.jax.dev/en/latest/debugging.html>; "JAX debugging
  flags" <https://docs.jax.dev/en/latest/debugging/flags.html>; "Profiling computation"
  <https://docs.jax.dev/en/latest/profiling.html>; "Device memory profiling"
  <https://docs.jax.dev/en/latest/device_memory_profiling.html>.
- jaxtyping, "Runtime type checking" <https://docs.kidger.site/jaxtyping/api/runtime-type-checking/>; beartype
  <https://beartype.readthedocs.io/>; Equinox, "Runtime errors" <https://docs.kidger.site/equinox/api/errors/>;
  chex <https://chex.readthedocs.io/>.
- Hypothesis <https://hypothesis.readthedocs.io/>; pytest <https://docs.pytest.org/en/stable/>; coverage.py
  <https://coverage.readthedocs.io/>; Pydantic, "Configuration" (the `extra` setting)
  <https://docs.pydantic.dev/latest/concepts/config/>.
- py-spy <https://github.com/benfred/py-spy> (its README explains the `SYS_PTRACE` capability needed inside
  Docker); Ray Dashboard <https://docs.ray.io/en/latest/ray-observability/getting-started.html>.
- Paper (free): Wilson, Aruliah, Brown, Chue Hong et al., "Best Practices for Scientific Computing", *PLoS
  Biology* 12(1): e1001745, 2014, doi:10.1371/journal.pbio.1001745.
- ToOp: "Running tests" in the course fork's `docker/README.md` (bytecode cache shared with the host,
  `--dist loadgroup`, `--max-worker-restart=0`); "Debugging Tips" in `.github/copilot-instructions.md`;
  `docs/wall_of_shame.md` (known shortcuts and unfixed bugs).

## Worked example

A property-based test states an invariant and lets Hypothesis search for a counterexample. Here the invariant is
"N-1 flows from LODFs equal a DC re-solve without the outaged branch", on random meshed grids (a ring has no
bridges, chords are added at random):

```python
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

def ptdf_matrix(fr, to, x, n_bus, slack=0):
    A = np.zeros((len(fr), n_bus)); A[np.arange(len(fr)), fr] = 1; A[np.arange(len(fr)), to] = -1
    Bf = A / x[:, None]; B = A.T @ Bf; keep = np.arange(n_bus) != slack
    ptdf = np.zeros_like(A); ptdf[:, keep] = np.linalg.solve(B[np.ix_(keep, keep)], Bf[:, keep].T).T
    return ptdf

@st.composite
def meshed_grids(draw, max_bus=8):
    n = draw(st.integers(3, max_bus))
    ring = [(i, (i + 1) % n) for i in range(n)]
    chords = draw(st.lists(st.tuples(st.integers(0, n - 1), st.integers(0, n - 1))
                           .filter(lambda e: e[0] != e[1]), max_size=4))
    fr, to = map(np.array, zip(*(ring + chords)))
    x = draw(arrays(np.float64, len(fr), elements=st.floats(0.01, 1.0)))
    p = draw(arrays(np.float64, n, elements=st.floats(-1.0, 1.0)))
    return fr, to, x, p

@settings(max_examples=200, deadline=None)
@given(meshed_grids())
def test_lodf_matches_resolve(grid):
    fr, to, x, p = grid
    n = len(p); ptdf = ptdf_matrix(fr, to, x, n); f0 = ptdf @ p
    for b in range(len(fr)):
        col = ptdf[:, fr[b]] - ptdf[:, to[b]]
        lodf = col / (1.0 - col[b])
        lodf[b] = -1.0                                   # the outaged branch loses all its flow
        keep = np.arange(len(fr)) != b
        f_ref = np.zeros(len(fr)); f_ref[keep] = ptdf_matrix(fr[keep], to[keep], x[keep], n) @ p
        np.testing.assert_allclose(f0 + lodf * f0[b], f_ref, atol=1e-8)
```

Delete the line `lodf[b] = -1.0` and Hypothesis immediately reports a failing grid: the formula alone gives
$\mathrm{LODF}_{b,b} = c_b/(1-c_b) \neq -1$, which is why both `numpy_reference.py` and `dc_solver/jax/lodf.py`
overwrite the diagonal explicitly. Run it in the ToOp container with `uv run --with hypothesis pytest <file>`.

## Lab

!!! example "Lab: find the planted bug"

    **Tools:** the ToOp container, pytest, Hypothesis (`uv run --with hypothesis ...`), Jupyter + NumPy + matplotlib,
    JAX debug flags and profiler, py-spy, git; pandapower or pypowsybl DC power flow as an oracle.

    **Instructor preparation.** Each team receives a ToOp branch with one undisclosed planted bug, for example:
    (a) a sign flip in the phase-shift columns (`dc_solver/preprocess/helpers/psdf.py`); (b) a float32 conversion
    in preprocessing (for example skipping `jax_enable_x64` in `dc_solver/preprocess/convert_to_jax.py`); (c) a
    wrong slack index in the PTDF computation (`dc_solver/preprocess/helpers/ptdf.py`); (d) a driver script that
    builds a new aggregation `lambda` on every call to `run_solver_symmetric`. The real stale-limits behavior of
    `run_solver` from the [Chapter 32](ch32-jax-in-toop.md) lab can serve as a fifth, unplanted case.

    1. **Reproduce.** Run `uv run pytest packages/dc_solver_pkg/tests/jax -q -m "not performance"` and the
       steps of `notebooks/example1_dc_loadflow_example.ipynb`; record every symptom (failing tests, changed
       overload energy, run time, warnings). Reading the branch diff is not allowed.
    2. **Localize.** Compare intermediate arrays with independent oracles: your own NumPy PTDF/LODF/PSDF code,
       `numpy_reference.py`, and a pandapower or pypowsybl DC power flow of the same grid. Use `jax.debug.print`,
       `jax_debug_nans` and, for performance symptoms, a JAX profiler trace or `py-spy record`. Narrow the fault to
       one function.
    3. **Test first.** Write a test that fails on the buggy branch: an oracle test, a tolerance test with a justified
       tolerance, or a Hypothesis property. Explain why ToOp's existing tests did or did not catch the bug (for
       example, absolute-value comparisons in `test_loadflows_match.py` cannot see a sign flip).
    4. **Fix and report.** Fix the bug, show the new test passing and the coverage of the changed lines, and write a
       one-page report: symptom, root cause, why it escaped, test added.

## Exercises

1. *(Jupyter: NumPy)* Scale all branch reactances of IEEE 118 by 10 and by 0.1. Which of ToOp's three singularity
   thresholds (LODF, BSDF, determinant) can change its decision for the same topology, and which cannot? Support
   the answer with the dimensions of each denominator and a numerical experiment.
2. *(Jupyter: JAX)* Compute N-1 flows under `jit` for a grid with a radial branch (a bridge). Run once normally,
   once with `jax.config.update("jax_debug_nans", True)`, and once with `jax.debug.print` of the denominator. Then
   call `calc_lodf` from `numpy_reference.py` on the same case. Compare the three behaviors and argue which one a
   batched solver should have.
3. *(Jupyter: Pydantic)* Construct `PreprocessParameters(enable_bb_outage=True)` and inspect `model_dump()`. Write a
   pytest helper that fails when a model silently drops a keyword, and use it to list which parameter models in
   `interfaces/messages/` and `optimizer/interfaces/messages/` accept unknown keywords.
4. *(pytest + Hypothesis)* Using your BSDF code from [Chapter 18](../part-3-power-systems/ch18-sensitivity-factors-2.md),
   write two properties on random meshed grids with one split substation: the BSDF-updated PTDF equals the PTDF
   rebuilt from the split network, and the flows satisfy Kirchhoff's current law at both busbars. Report the
   smallest counterexample Hypothesis finds when you introduce an off-by-one in the busbar assignment.
5. *(Octave)* With MATPOWER, compute the PTDF of `case118` with `makePTDF` and the LODF with `makeLODF`, once from
   the double-precision PTDF and once from `single(full(PTDF))`. Plot the largest LODF error per outage against the
   LODF denominator, and discuss whether a threshold of `1e-11` makes sense for float32 data.
6. *(ToOp container: JAX profiler + py-spy)* Write a script that repeats
   `notebooks/example1_dc_loadflow_example.ipynb` and then calls
   `run_solver(default_topology(sc, batch_size=4), None, None, di, sc)` twice (`sc` and `di` are the two halves of
   `static_information`). Wrap the second `run_solver` call in `jax.profiler.trace` and open the trace in
   Perfetto; then run the same script under `py-spy record -o profile.svg` (the container needs `SYS_PTRACE`).
   Estimate the share of compilation, device computation and Python overhead in the first and the second call.
