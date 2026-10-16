# Chapter 30 – Solutions

Solutions of the Lab and the Exercises of [Chapter 30](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`two_libraries.py`](src/two_libraries.py) (the worked example),
[`incidence.py`](src/incidence.py), [`sparse_vs_dense.py`](src/sparse_vs_dense.py),
[`ieee14_flows.py`](src/ieee14_flows.py), [`case57_ac_dc.py`](src/case57_ac_dc.py) and
[`loading_query.py`](src/loading_query.py), each with a `test_*.py` next to it. Every script prints the numbers
quoted below when run from the `src` folder (`python incidence.py`, and so on). The numbers come from the course
environment (pandapower 3.5, pypowsybl 1.16, NumPy 2.4, SciPy 1.18, Python 3.14 on Windows); the Polars parts
were run with ToOp's environment (Polars 1.42, ToOp commit `ba1874e`), which is the one that has Polars. Lab
steps 1–2 and Exercise 6 quote the output of commands run in the ToOp container, given with each solution.

## Lab

1. **Environment.** The code lives in `/app`, a bind mount of your checkout on the host, so a file edited on
   Windows is the file the container imports, with no copy and no rebuild (the six `packages/*` are installed
   editable). The environment lives in `/opt/venv`, on the named volume `toop-venv`, because the Dockerfile sets
   `UV_PROJECT_ENVIRONMENT=/opt/venv` (and puts `/opt/venv/bin` first on `PATH`). They are kept apart because a
   Linux virtual environment under `/app/.venv` would collide with a `.venv` created natively on Windows and
   would push thousands of small files across the WSL2 file-system bridge on every import; on the named volume
   it stays on the Linux side and survives `docker compose down` (`docker compose down -v` deletes it, which is
   the documented repair when it gets into a bad state). The facts are from the course fork's `docker/README.md`
   ("Why the virtualenv is not in the repository folder") and `docker/Dockerfile`, and the container confirms
   them (see [Exercise 6](#exercise-6)): `/opt/venv/bin/python` is a symbolic link to the image's
   `/usr/local/bin/python3.11`, so the environment holds the packages and borrows the interpreter, and the
   environment's 3.0 GB (`du -sh /opt/venv`) never touch the Windows side.
2. **Tests.** The threshold is in the `[report]` section of `.coveragerc`: `fail_under = 90`, with
   `show_missing = true` (the report lists the uncovered line numbers) and `sort = cover` (the report is sorted
   by coverage, so the least-covered modules are at the top of the table and the three you are asked for are its
   first three rows). The plain run is `uv run pytest packages/interfaces_pkg/tests -q`; the CI-style run adds
   `--cov=packages/interfaces_pkg/src --cov-config=.coveragerc`, with `UV_PROJECT=packages/interfaces_pkg` so
   that `uv run` resolves the package's own project.

    Both runs were made in the course image `toop-course:cpu` (built from the course fork's `docker/Dockerfile`),
    with a checkout of ToOp commit `ba1874e` bind-mounted read-only at `/app`, JAX on the CPU, and Docker
    Desktop's VM at 16 CPUs and 13.6 GB, shared with other ToOp containers during the runs. From Git Bash:

    ```bash
    cd ToOp
    MSYS_NO_PATHCONV=1 docker run --rm --shm-size=2gb -e COVERAGE_FILE=/tmp/.coverage \
      -v "$(pwd -W):/app:ro" -w /app toop-course:cpu \
      bash -c 'UV_PROJECT=packages/interfaces_pkg uv run pytest packages/interfaces_pkg/tests \
        --cov=packages/interfaces_pkg/src --cov-config=.coveragerc --junit-xml=/tmp/test-report.xml'
    ```

    The read-only mount, `COVERAGE_FILE` and `--junit-xml=/tmp/...` only keep the checkout untouched: ToOp's
    `addopts` writes `test-report.xml` and coverage writes `.coverage` into the current folder, `/app`. With the
    compose file you type the commands of the lab as they are. Every fresh container started from the image's
    own `/opt/venv` first rebuilds the seven editable ToOp packages (`Prepared 7 packages in 26.40s`), because
    their version string comes from the git commit of the mounted checkout, not of the build.

    - *Plain run:* `160 passed, 4 warnings in 195.35s`. One test takes half of it:
      `test_stored_action_set_performance.py::test_stored_action_set_large_performance` (marked `performance`)
      ran 93.29 s; the next slowest, in `test_loadflow_result_helpers_new.py`, 8.19 s.
    - *CI-style run:* `160 passed, 4 warnings in 229.42s`, ending `Required test coverage of 90.0% reached.
      Total coverage: 92.95%` (2596 statements, 183 missed). The top of the table:

        | Module (under `toop_engine_interfaces/`) | Stmts | Miss | Cover |
        |---|---|---|---|
        | `folder_structure.py` | 6 | 6 | 0 % |
        | `messages/protobuf_message_factory.py` | 8 | 8 | 0 % |
        | `network_masks.py` | 68 | 68 | 0 % |
        | `switch_update_schema.py` | 7 | 7 | 0 % |
        | `asset_topology/topology_conversion.py` | 156 | 30 | 81 % |
        | `asset_topology/runtime_topology.py` | 244 | 21 | 91 % |

        Four modules tie at 0 %: not even imported during the run, or their `import` and `def` lines would
        count as covered (coverage lists ties by name). So "the three least covered" are the first three rows, with `switch_update_schema.py` just as
        uncovered; the first partly tested module is `topology_conversion.py`. The message modules are at
        100 %, and so is `filesystem_helper.py`, the `fsspec` wrapper.
    - *A timeout that depends on the machine.* Under coverage the performance test took 126.93 s instead of
      93 s (and 116.93 s against 94.91 s when run alone, back to back). It carries
      `@pytest.mark.timeout(180)`, and in a first coverage run, made while three ToOp pipeline containers kept
      about five CPUs busy, it passed that limit: `Failed: Timeout (>180.0s) from pytest-timeout`, `1 failed,
      159 passed in 319.80s`, with the same 92.95 % coverage but exit code 1. A wall-clock limit tests the
      machine as much as the code; on a busy machine deselect that test with `-m "not performance"`. ToOp's CI
      runs the same command with `-n auto --dist loadgroup` (pytest-xdist,
      `.github/actions/run-package-tests/action.yaml`).
3. **Reference implementation.** `packages/dc_solver_pkg/tests/numpy_reference.py` is one module of fifteen
   functions, eight of them private (leading underscore):

    | Group | Functions |
    |---|---|
    | BSDF building blocks | `_validate_assignment`, `_get_local_branch_data`, `_compute_ptdf_difference`, `_adjust_ptdf_for_slack`, `_compute_bsdf_denominator`, `_compute_pedf_terms`, `_compute_bsdf_vector`, `_update_network_after_split` |
    | BSDF entry points | `get_bsdf_branch_indices`, `calc_bsdf`, `extract_bsdf_data`, `compute_bus_splits` |
    | N-1 | `calc_lodf`, `contingency_analysis`, `run_solver` |

    - *Shape annotations.* Every array argument and return value carries a jaxtyping type with a shape
      string: `Float[np.ndarray, " n_branches n_nodes"]` for the PTDF, `Int[np.ndarray, " n_branches"]` for
      `from_node` and `to_node`, `Bool[np.ndarray, " n_branches_at_node"]` for the busbar assignment, and tuples
      of them for multiple return values. The names in the strings (`n_branches`, `branches_from_a`, ...) are
      documentation here; the same strings are checked at run time in the JAX code when beartype is on
      ([Chapter 31](../ch31-jax-gpu-fundamentals/index.md), Exercise 6).
    - *Docstrings.* NumPy style: a summary line, then `Parameters`, `Returns` and `Raises` sections with
      `name : type` headers, not the Google style of this course's code.
    - *Fancy indexing.* Integer arrays pick columns in `ptdf[:, i] - ptdf[:, j]` (`calc_lodf`) and rows in
      `ptdf[branch_to_a, :]` (`_compute_ptdf_difference`); boolean masks select branches in
      `branches_locally[~assignment & branch_direction_locally]` (`get_bsdf_branch_indices`); an in-place
      fancy-indexed update `nom[branch_from_a] += suscept_from_a` in `_compute_bsdf_vector`; a slice of a
      boolean vector `topo_vect[topo_vect_index : topo_vect_index + len(branches_at_node)]` in
      `extract_bsdf_data`.
    - *Broadcasting.* `ptdf[:, node_from_a] - ptdf_bus_a` subtracts a `(n_branches, 1)` column from a
      `(n_branches, k)` block (`_compute_pedf_terms`); `suscept_from_a * pedf_from` multiplies a `(k,)` vector
      into every row of a `(n_branches, k)` matrix; `np.outer(bsdf, ptdf_diff)` is the rank-one PTDF update of
      [Chapter 18](../../part-3-power-systems/ch18-sensitivity-factors-2/index.md); and
      `n_0_flows + lodf * n_0_flows[branch_to_outage]` is the LODF update of
      [Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md).
    - *Errors.* `_validate_assignment` raises `AssertionError` when a busbar gets no branch (and
      `get_bsdf_branch_indices` and `extract_bsdf_data` use bare `assert` statements for length checks);
      `_compute_bsdf_denominator` raises `ValueError` when $|\text{denom}| < 10^{-5}$ (the split would island
      the grid) and `calc_lodf` raises `ValueError` when $|\text{denom}| < 10^{-11}$ (a bridge). Only the
      second is caught: `contingency_analysis` wraps `calc_lodf` in `try/except ValueError`, stores a zero row
      and `False` in the success array, and goes on, which is exactly the "flag instead of exception" habit the
      JAX version must adopt ([Chapter 31](../ch31-jax-gpu-fundamentals/index.md)). A BSDF failure in
      `compute_bus_splits` propagates and stops `run_solver`.
    - *The test that imports it.* `packages/dc_solver_pkg/tests/jax/test_lodf.py`:
      `from tests.numpy_reference import calc_lodf as calc_lodf_ref`, compared against
      `toop_engine_dc_solver.jax.lodf.calc_lodf`.
    - *Two things to notice as a reader.* The module docstring mentions a `bsdf_check_split` function "also used
      to pre-filter the action set" that no longer exists in the file; and the inline comments of `calc_lodf`
      speak of shapes `(M,)` and `(E, M)` while its signature takes a single `int`, a trace of an earlier
      vectorized version. Reference code drifts too.
4. **Black-box power flow.** `case57_ac_dc.side_by_side()` calls the worked example's `pandapower_case57()`
   and puts the AC and DC `p_from_mw` of the 63 lines of case57 (the other 17 branches are transformers, in
   `res_trafo`) in one frame; `to_polars()` converts it with `pl.from_pandas` and `histogram()` draws the
   difference. The AC − DC difference has mean 0.26 MW, standard deviation 2.62 MW, and ranges from −4.51 MW to
   10.18 MW; the largest gap is on line 14 (AC 149.6 MW, DC 139.4 MW). With ten bins the histogram counts are
   `[4, 7, 16, 26, 3, 3, 1, 0, 2, 1]` between −4.5 MW and 10.2 MW: a tall peak just above zero (42 of 63 lines
   within ±1.4 MW) and a thin right tail, the heavily loaded lines where losses, which the DC model drops, are
   largest. The Polars frame has shape `(63, 4)` (the line index becomes a column) and the same sums
   (`test_case57_ac_dc.py`, run with ToOp's interpreter). To commit the notebook with its outputs cleared, from
   your course checkout:

    ```bash
    git checkout -b docs/case57-ac-dc
    jupyter nbconvert --clear-output --inplace case57_ac_dc.ipynb   # or let the nbstripout hook do it
    git add case57_ac_dc.ipynb
    git commit -m "docs: add case57 AC vs DC notebook"
    ```

    *Not run here*: the commit is yours to make; the repository's `nbstripout` pre-commit hook strips outputs
    on commit in any case.

    ```python title="case57_ac_dc.py"
    --8<-- "docs/part-5-computing/ch30-scientific-python/src/case57_ac_dc.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch30-scientific-python/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Two fancy-indexed assignments build the whole matrix: `a[rows, from_bus] = 1.0` and `a[rows, to_bus] = -1.0`,
with `rows = np.arange(n_branch)`. Each row has exactly one $+1$ and one $-1$, so it sums to zero, and
$A^\top A$ is the Laplacian ([Chapter 2](../../part-1-math/ch02-linear-algebra/index.md)): for the four-bus,
five-branch grid of the Chapter 31 worked example its diagonal is the degree sequence $(2, 3, 3, 2)$. The test
is parametrized over a three-bus path, a triangle and that four-bus grid, and also checks the Laplacian of the
triangle and the two `ValueError`s. `uv run pytest docs/part-5-computing/ch30-scientific-python/src/test_incidence.py`
reports 5 passed (three parametrized cases plus two tests).

```python title="incidence.py"
--8<-- "docs/part-5-computing/ch30-scientific-python/src/incidence.py"
```

```python title="test_incidence.py"
--8<-- "docs/part-5-computing/ch30-scientific-python/src/test_incidence.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch30-scientific-python/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

`random_symmetric(n)` draws `scipy.sparse.random(n, n, density=0.02)`, symmetrizes it with $M + M^\top$ and adds
$2n I$ so the matrix is strictly diagonally dominant, hence never singular; at $n = 200$ it has 1 766 nonzeros of
40 000. The two solvers agree to $3.9 \times 10^{-18}$. `timings()` takes the best of three `timeit` runs of
`spsolve` and of `np.linalg.solve(a.toarray(), b)` (the conversion is part of the dense route). On the machine
that wrote this (CPU timings, two runs, so you see how much they move):

| $n$ | `spsolve` [ms] | dense solve [ms] | dense / sparse |
|---|---|---|---|
| 200 | 1.0–1.1 | 36–66 | 34–67 |
| 400 | 5.7–6.5 | 162–192 | 25–33 |
| 800 | 33–36 | 362–533 | 10–16 |
| 1600 | 276–296 | 542–988 | 2.0–3.3 |
| 3200 | 2 077–2 093 | 935–1 076 | 0.5 |

On log-log axes the dense curve is a line of slope about 3 ($n^3$ flops) once $n$ is large enough for the
BLAS overhead to be negligible, while the sparse curve is steeper, close to slope 3 as well between 800 and 3200:
the LU factors of a *random* sparse matrix fill in almost completely, so the sparse solver ends up doing dense
work with sparse bookkeeping and the dense solver overtakes it near $n \approx 2000$–3000. Grid matrices are
different: a power system's $B$ matrix is close to planar, and a fill-reducing ordering keeps its factors
sparse, which is why SciPy sparse (and ToOp's preprocessing) stay far ahead on real grids. The lesson of the
plot is that "sparse" is a property of the pattern, not of the count of zeros.

```python title="sparse_vs_dense.py"
--8<-- "docs/part-5-computing/ch30-scientific-python/src/sparse_vs_dense.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch30-scientific-python/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 3

`line_flows()` runs `run_ac`, copies `get_lines()['p1']`, runs `run_dc` and reads the same column again (the DC
run overwrites it, which is why the copy comes first), then sorts by $|p_1|$ of the AC run. Both load flows
converge. The 17 lines, in MW at side 1:

| Line | AC $p_1$ | DC $p_1$ | Line | AC $p_1$ | DC $p_1$ |
|---|---|---|---|---|---|
| L1-2-1 | 156.88 | 153.45 | L6-13-1 | 17.75 | 17.26 |
| L1-5-1 | 75.51 | 72.25 | L9-14-1 | 9.43 | 9.63 |
| L2-3-1 | 73.24 | 69.83 | L6-12-1 | 7.79 | 7.61 |
| L4-5-1 | −61.16 | −62.28 | L6-11-1 | 7.35 | 6.75 |
| L2-4-1 | 56.13 | 54.77 | L13-14-1 | 5.64 | 5.27 |
| L2-5-1 | 41.52 | 40.45 | L9-10-1 | 5.23 | 5.75 |
| L7-9-1 | 28.07 | 28.34 | L10-11-1 | −3.79 | −3.25 |
| L3-4-1 | −23.29 | −24.37 | L12-13-1 | 1.61 | 1.51 |
| | | | L7-8-1 | 0.00 | −0.00 |

The largest gap is 3.43 MW on L1-2-1 and the mean gap 1.06 MW. The DC flows are a little smaller on the lines
leaving the slack because the AC solution also carries the losses; L7-8-1 carries no active power in either
model, since bus 8 holds only a synchronous condenser. `bar_chart()` draws the two series as paired bars
(AC left, DC right of each line label, in the sorted order), so the figure reads as a monotone staircase with a
thin second staircase hugging it.

```python title="ieee14_flows.py"
--8<-- "docs/part-5-computing/ch30-scientific-python/src/ieee14_flows.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch30-scientific-python/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`res_line_case57()` joins `from_bus` and `to_bus` from `net.line` onto `net.res_line` (63 lines). The pandas
version is eager, two statements: a boolean mask with the median, then `groupby('from_bus')['loading_percent'].mean()`.
The Polars version is one lazy expression chain: `pl.from_pandas(...).lazy().filter(pl.col('loading_percent') >
pl.col('loading_percent').median()).group_by('from_bus').agg(pl.col('loading_percent').mean()).sort('from_bus').collect()`.
Nothing is computed until `collect()`, the median is an expression inside the filter rather than a value
computed beforehand, and `group_by` does not sort its output (pandas does), hence the explicit `sort`. Results:
the median loading is 0.17 %, 31 of the 63 lines are above it, they leave from 23 distinct buses, and the five
highest means are from bus 7 (1.78 %), 0 (1.13 %), 1 (0.99 %), 6 (0.84 %) and 13 (0.67 %). The Polars frame
holds the same 23 rows to floating-point precision (`test_loading_query.py`, run with ToOp's interpreter;
the `from_bus` column comes back as `u32`, pandas keeps it as `int64`). The percentages are tiny because the
MATPOWER case57 ships no line ratings and pandapower fills `max_i_ka` with a very large value: the exercise is
about the query, and the ranking is what the numbers mean.

```python title="loading_query.py"
--8<-- "docs/part-5-computing/ch30-scientific-python/src/loading_query.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-5-computing/ch30-scientific-python/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

A session that does what the statement asks (*not run here*: it mutates a repository of your own):

```bash
mkdir cc-demo && cd cc-demo && git init -b main
echo "print('hi')" > app.py && git add app.py && git commit -m "feat: add greeting script"
echo "print('hello')" > app.py && git commit -am "fix: spell the greeting out"
echo "# demo" > README.md && git add README.md && git commit -m "docs: add readme"
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
EOF
pre-commit install
printf 'x = 1   \n' > bad.py && git add bad.py .pre-commit-config.yaml && git commit -m "feat: add bad file"
# trailing-whitespace....Failed  -  hook id: trailing-whitespace  -  files were modified by this hook
```

The hook strips the spaces and fails the commit; `git add bad.py` and commit again to get it through. ToOp's
`validate_pr_title.yaml` installs commitizen and runs `cz check --message "$PR_TITLE"`, and the course
environment has the same tool, so the question can be settled without a pull request:

| Title | `python -m commitizen check --message ...` |
|---|---|
| `Fix/some slug` | rejected: does not match the pattern `(build|bump|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\(\S+\))?!?: ...` |
| `fix: handle empty action set` | `Commit validation: successful!` |
| `update docs` | rejected, same pattern (no type, no colon) |

`Fix/some slug` is what GitHub proposes for a branch named `fix/some-slug` with more than one commit (type
capitalised, colon dropped), so a branch of several commits needs its title typed by hand, or the squash merge
lands a subject the convention rejects.

## Exercise 6

Run in the course image `toop-course:cpu` (built from the course fork's `docker/Dockerfile`) with ToOp commit
`ba1874e` mounted at `/app`. Both commands print `/opt/venv/bin/python`: the Dockerfile sets
`UV_PROJECT_ENVIRONMENT=/opt/venv` *and* puts `/opt/venv/bin` first on `PATH`, so the shell's `python` and
`uv run`'s interpreter are the same file. `which -a python` also lists `/usr/local/bin/python`, the image's own
interpreter, which is what a stock `python:3.11-slim` image without that `PATH` change would answer; and
`readlink -f /opt/venv/bin/python` gives `/usr/local/bin/python3.11`: the environment is a folder of packages
around the image's interpreter.

To see what a restart keeps without touching the compose project, the same experiment can be made with two
`docker run --rm` containers (each one deleted on exit, as `docker compose down` deletes its container) that
share a named volume at `/opt/venv`, as the compose file mounts `toop-venv`. From Git Bash, in the ToOp
checkout:

```bash
docker volume create try-venv
for i in 1 2; do
  MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/app:ro" -v try-venv:/opt/venv -w /app toop-course:cpu \
    bash -c 'ls /opt/venv/marker.txt /root/marker.txt; date > /opt/venv/marker.txt; date > /root/marker.txt'
done
docker volume rm try-venv
```

The first container starts with an empty volume, which Docker seeds from the image's `/opt/venv`; its
entrypoint's `uv sync` rebuilds the seven editable ToOp packages (`Prepared 7 packages in 29.85s`) and `ls`
finds neither marker. The second container prints `Checked 312 packages in 60ms` (the rebuilt packages are on
the volume) and finds `/opt/venv/marker.txt` but not `/root/marker.txt`. So, after `docker compose down` and
`docker compose up -d`:

- `/opt/venv` survives: it is the named volume `toop-venv`, which `down` leaves alone (only `down -v` removes
  volumes). The entrypoint still runs `uv sync` at every start, but it finds nothing to do (60 ms above).
- `/app` survives for a different reason: it is a bind mount, the files live on the host.
- Everything written elsewhere in the container's file system (a `pip install` into the image's Python, a file
  under `/root` or `/tmp`, an `apt-get install`) lives in the container's writable layer, which `down` deletes
  together with the container; `up -d` starts a fresh container from the image, so that state is gone.

The rule: state you want to keep goes in a volume or a bind mount, and the image is rebuilt, never patched in
place.
