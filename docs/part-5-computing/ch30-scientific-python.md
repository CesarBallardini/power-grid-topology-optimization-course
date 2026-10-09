# Chapter 30. Scientific Python

**Weeks:** 3.5 · **Semester:** 1 · **Prerequisites:** none

!!! info "Why it matters for ToOp"

    Every lab in this plan is in Python, and ToOp's preprocessing, backends and data
    handling use NumPy, SciPy sparse, NetworkX, pandas and Polars with full type annotations.

    - **Grid backends.** ToOp reads grids through two open-source power system libraries behind one abstract
      class: `BackendInterface` (`interfaces/backend.py`) with `PandaPowerBackend`
      (`dc_solver/preprocess/pandapower/pandapower_backend.py`) and `PowsyblBackend`
      (`dc_solver/preprocess/powsybl/powsybl_backend.py`). The AC stage calls pandapower or pypowsybl and returns
      Polars or pandas frames (`contingency/ac_loadflow_service/ac_loadflow_service.py`).
    - **Tests.** Six packages under `packages/`, each with a pytest suite and a 90% coverage gate (`.coveragerc`).
      Notebooks in `notebooks/` are executed as tests.
    - **Git workflow.** Trunk-based development with squash merges; pull request titles must follow Conventional
      Commits (`.github/workflows/validate_pr_title.yaml`), and local checks run through pre-commit
      (`.pre-commit-config.yaml`). See `docs/contribution_guide.md`.
    - **Environments.** Dependencies are managed by `uv` (`uv.lock`); on Windows ToOp runs in a Linux container
      ([Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md)).

## Topics

- Python: functions, classes, dataclasses, type hints, modules and packages.
- NumPy: arrays, broadcasting, fancy indexing, `einsum`; SciPy sparse matrices and solvers.
- NetworkX graphs; pandas and Polars data frames (the same table in both; lazy vs eager evaluation in Polars).
- matplotlib: line plots, histograms, heat maps of a matrix, saving figures for a report.
- Jupyter: notebooks vs scripts, kernels, restart-and-run-all as a reproducibility check, clearing outputs before a
  commit.
- Power system libraries as black boxes: load a test grid in pandapower and pypowsybl, inspect element tables
  (buses, lines, transformers, generators, loads) as data frames, run an AC and a DC power flow, read the result
  tables. The theory comes later ([Chapter 15](../part-3-power-systems/ch15-ac-power-flow.md),
  [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md)).
- git: commits, branches, merges, remotes, forks and pull requests; Conventional Commits messages; pre-commit hooks.
- Virtual environments and `uv` (`uv sync`, `uv run`, lock files); testing with pytest (fixtures, parametrization,
  markers, coverage).
- Containers: image vs container, bind mounts and named volumes, `docker compose`, and why ToOp's Windows path is
  a Linux container (revisits [Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md)).

## Learning outcomes

- Write a type-annotated NumPy function with a pytest test, and run it from `uv`.
- Load a test grid in pandapower and pypowsybl, run a power flow, and turn a result table into a pandas or Polars
  frame and a matplotlib figure.
- Use git to branch, commit with a Conventional Commits message, and prepare a pull request that passes
  pre-commit hooks.
- Start a container, run a command in it, and explain where the code, the environment and the data live.

## Resources

- Courses: MIT OCW 6.100L <https://ocw.mit.edu/courses/6-100l-introduction-to-cs-and-programming-using-python-fall-2022/>;
  Software Carpentry *Programming with Python* <https://swcarpentry.github.io/python-novice-inflammation/> and
  *Version Control with Git* <https://swcarpentry.github.io/git-novice/>;
  MIT *The Missing Semester* (shell, git) <https://missing.csail.mit.edu/>.
- Free texts:
  - Downey, *Think Python*, 3rd ed.: <https://allendowney.github.io/ThinkPython/>.
  - VanderPlas, *Python Data Science Handbook*, Ch. 2 NumPy and Ch. 3 pandas:
    <https://jakevdp.github.io/PythonDataScienceHandbook/>.
  - McKinney, *Python for Data Analysis*, 3rd ed., open web edition: <https://wesmckinney.com/book/>.
  - *Scientific Python Lectures*: <https://lectures.scientific-python.org/>.
  - Chacon & Straub, *Pro Git*: <https://git-scm.com/book/en/v2> (official Spanish translation, FREE:
    <https://git-scm.com/book/es/v2>).
- Documentation:
  - NumPy, SciPy sparse, NetworkX tutorial, Polars <https://docs.pola.rs/>, pytest <https://docs.pytest.org/en/stable/>,
    coverage.py <https://coverage.readthedocs.io/>, uv <https://docs.astral.sh/uv/>.
  - pandas getting started <https://pandas.pydata.org/docs/getting_started/index.html>; matplotlib getting started
    <https://matplotlib.org/stable/users/getting_started/>; Jupyter <https://docs.jupyter.org/> and JupyterLab
    <https://jupyterlab.readthedocs.io/>.
  - pandapower documentation <https://pandapower.readthedocs.io/> and tutorials
    <https://github.com/e2nIEE/pandapower/tree/develop/tutorials>; pypowsybl documentation
    <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/>, in particular "Running a load flow"
    <https://powsybl.readthedocs.io/projects/pypowsybl/en/stable/user_guide/loadflow.html>.
  - Conventional Commits 1.0.0 <https://www.conventionalcommits.org/en/v1.0.0/>; pre-commit <https://pre-commit.com/>;
    commitizen (used by ToOp to validate titles) <https://commitizen-tools.github.io/commitizen/>; GitHub, "Fork a
    repository" <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo>.
  - Docker get started <https://docs.docker.com/get-started/>; Docker Compose <https://docs.docker.com/compose/>.
- Textbooks: Downey, *Think Python* (2012) **BORROW** [`thinkpython0000down`](https://archive.org/details/thinkpython0000down);
  McKinney, *Python for Data Analysis* 1st ed. **BORROW** [`pythonfordataana0000mcki`](https://archive.org/details/pythonfordataana0000mcki)
  (ES: *Python para análisis de datos*, Anaya, 2023);
  Ramalho, *Fluent Python* (2015) **BORROW** [`fluentpython0000rama`](https://archive.org/details/fluentpython0000rama).

## Worked example

Both libraries ship test grids and store every element type as a data frame. Treat the solvers as black boxes
for now; only the data handling matters here (checked with pandapower 3.1 and pypowsybl 1.15, the versions in
ToOp's lock file).

```python
import pandapower as pp
import pandapower.networks as pn
import pypowsybl as pypo

net = pn.case57()                         # IEEE 57-bus case in pandapower
print(len(net.bus), len(net.line), len(net.trafo))   # element tables are pandas DataFrames
pp.runpp(net)                             # AC power flow; results land in net.res_line, net.res_bus, ...
ac_p = net.res_line.p_from_mw.copy()
pp.rundcpp(net)                           # DC power flow overwrites the same result tables
dc_p = net.res_line.p_from_mw

n = pypo.network.create_ieee14()          # IEEE 14-bus case in pypowsybl
result = pypo.loadflow.run_ac(n)
print(result[0].status)                   # ComponentStatus.CONVERGED
lines = n.get_lines()[["p1", "i1"]]       # results are columns of the element tables
```

Note the two different styles: pandapower writes results into `res_*` tables of a mutable `net` object, while
pypowsybl updates the network and exposes results as columns of `get_lines()`, `get_buses()` and so on.

## Lab

!!! example "Lab: install ToOp, run a package's tests, read a reference implementation"

    **Tools:** the ToOp container from [Chapter 0](../part-0-orientation/ch00-orientation-and-setup.md) (or, on
    Linux, macOS or WSL, a host install with `uv sync --all-groups`); pytest; git; Jupyter + NumPy + pandas +
    matplotlib; pandapower and pypowsybl.

    1. **Environment.** Use the Chapter 0 container, or install on a host with `uv sync --all-groups`. Explain in
       two sentences where the code (`/app`, a bind mount of your checkout) and the environment (`/opt/venv`, a
       named volume) live in the container, and why the image keeps them apart ("Why the virtualenv is not in the
       repository folder" in the course fork's `docker/README.md`).
    2. **Tests.** Run one package's suite, `uv run pytest packages/interfaces_pkg/tests -q`, then the CI-style run
       with coverage (one command):

        ```bash
        UV_PROJECT=packages/interfaces_pkg uv run pytest packages/interfaces_pkg/tests \
          --cov=packages/interfaces_pkg/src --cov-config=.coveragerc
        ```

        Find the coverage threshold in `.coveragerc` and the three least-covered modules in the report.
    3. **Reference implementation.** Read `packages/dc_solver_pkg/tests/numpy_reference.py` end to end as a piece
       of Python, not yet as mathematics: list the functions, the jaxtyping shape annotations
       (`Float[np.ndarray, " n_branches n_nodes"]`), the numpy-style docstrings, where fancy indexing and
       broadcasting are used, and which errors are raised. Find one test that imports it
       (`packages/dc_solver_pkg/tests/jax/test_lodf.py`). You will return to its mathematics in
       [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md).
    4. **Black-box power flow.** In a notebook, load pandapower's `case57`, run AC and DC power flows, put the two
       `p_from_mw` columns side by side in a pandas and a Polars frame, and plot a histogram of their difference
       with matplotlib. Save the notebook with outputs cleared, and commit it on a branch of your own course
       repository with a Conventional Commits message.

## Exercises

1. *(Jupyter: NumPy + pytest)* Write `incidence_matrix(from_bus, to_bus, n_bus) -> np.ndarray` with type hints,
   and a parametrized pytest test that checks each row sums to zero for three small graphs. Run it with
   `uv run pytest`.
2. *(Jupyter: NumPy + matplotlib)* Build a random 200×200 sparse symmetric matrix with SciPy, solve a linear
   system with `scipy.sparse.linalg.spsolve` and with dense `numpy.linalg.solve`, and plot the run time of both
   against matrix size.
3. *(Jupyter: pypowsybl + pandas + matplotlib)* Load `pypowsybl.network.create_ieee14()`, run an AC load flow, and
   plot active power `p1` of every line as a bar chart sorted by magnitude. Repeat with `run_dc` and put both
   series in one figure.
4. *(Jupyter: pandas + Polars)* Convert `net.res_line` of pandapower's `case57` to Polars, keep lines with
   `loading_percent` above the median, and compute the mean loading grouped by `from_bus`. Write the same query in
   pandas and compare the code.
5. *(git)* Create a repository, make three commits whose messages follow Conventional Commits (`feat:`, `fix:`,
   `docs:`), install one pre-commit hook (for example `trailing-whitespace`), and show a commit it rejects. Which
   title would ToOp's `validate_pr_title.yaml` reject: `Fix/some slug`, `fix: handle empty action set`, or
   `update docs`?
6. *(ToOp container)* Inside the container run `which python` and `uv run which python`, then `docker compose
   down` and `up -d` again. Which state survives the restart and why (named volume vs container layer)?
