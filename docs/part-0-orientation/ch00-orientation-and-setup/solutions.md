# Chapter 0 – Solutions

Solutions of the Lab and the Exercises of [Chapter 0](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`environment_check.py`](src/environment_check.py) runs anywhere JAX is
installed and prints the device report and the two timings; [`case57_toop.py`](src/case57_toop.py) runs inside
the ToOp container (or with ToOp's virtual environment) and does what `notebooks/example1_dc_loadflow_example.ipynb`
does, printing the file listing, the counts and the timings quoted below. The numbers were produced with ToOp
`ba1874e` on CPU (Windows 11, JAX 0.5.3 in ToOp's environment, JAX 0.11.2 in the course environment); the
container itself and the course JupyterHub were not run here, so the Docker commands are given as method plus
expected outcome.

## Lab

1. **Which device.** `python -c "import jax; print(jax.devices())"` printed `[CpuDevice(id=0)]` in both
   environments, and `environment_check.py` reports `backend cpu`, `kind CPU`, `x64 False` (ToOp switches
   `jax_enable_x64` on itself, see `case57_toop.py`). That is the expected answer for every lab of this book: the
   GPU is optional, and the chapter explains why it would not be faster on these grids. In the container the same
   line appears in the start banner (`docker compose logs toop-cpu`), with the cause named when the GPU image
   falls back to the CPU.

    ```python title="environment_check.py"
    --8<-- "docs/part-0-orientation/ch00-orientation-and-setup/src/environment_check.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-0-orientation/ch00-orientation-and-setup/src){ target="_blank" } to see or download all the files of this chapter.

2. **Two runs, two timings.** The notebook was not run as a notebook here; `case57_toop.py` runs the same calls
   (`case57_data_powsybl`, `load_grid`, `run_initial_loadflow`) as a script. Two runs of the script in two fresh
   processes took 15.0 s and 14.3 s of wall-clock time, a 5 % difference, and a third fresh run 16.7 s: the
   second run is not reliably faster. The reason is that a restarted kernel (or a new process) **compiles again**:
   JAX keeps compiled programs in memory, and a persistent compilation cache on disk is off unless
   `jax_compilation_cache_dir` is set. What the second run saves is only what the operating system cached
   (imported modules, the data files), a few hundred milliseconds. Compilation shows up *within* one session:
   inside the script, the first `run_initial_loadflow` took 0.67 s and the second 0.50 s (0.81 s and 0.54 s on
   another run); in `environment_check.py`, the first call of a small jitted solve took 237 ms and the second
   0.4 ms, so there the compilation is practically the whole first call. On a GPU the first call of a session
   also pays for the CUDA context, which is why the chapter asks to time twice *without* restarting in between.
   The notebook's "run all, restart, run all" therefore measures mostly disk caching, and the honest answer to
   "which part is compilation" is: almost none of the difference between the two notebook runs; the compilation is
   paid in both.
3. **The processed grid folder.** Written into a new folder, the generator `case57_data_powsybl(folder)` leaves
   the output of the **import stage**: `grid.xiidm` (the grid model in pypowsybl format), `masks/` (which lines,
   transformers, generators and substations the optimizer may touch), `initial_topology/` (the substations as
   busbars and couplers, read later by the AC stage), `loadflow_parameters.json` (the settings for the AC load
   flow) and `logs/`. After `load_grid`, the **DC preprocessing** has added `static_information.hdf5` (the arrays
   the GPU solver reads), `static_information_stats.json`, `action_set.json` and `action_set_diffs.hdf5` (every
   busbar split the optimizer may try, and the same actions as changes of the incidence) and
   `nminus1_definition.json` (the contingency list, reused by the AC validation). Nothing in the folder belongs to
   the **AC validation** stage yet: its results go to a snapshot folder (`optimizer_snapshots/`) once an
   optimization has run. `case57_toop.py` prints this listing before and after `load_grid` and the stage of each
   entry; `STAGE_OF` in the script is the table.

    ```python title="case57_toop.py"
    --8<-- "docs/part-0-orientation/ch00-orientation-and-setup/src/case57_toop.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-0-orientation/ch00-orientation-and-setup/src){ target="_blank" } to see or download all the files of this chapter.

4. **Terms to revisit.** There is no single answer. A list taken from the notebook and the folder of step 3 that
   the course explains later: *DC load flow* and *PTDF* ([Chapter 16](../../part-3-power-systems/ch16-dc-power-flow/index.md),
   [Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md)), *N-1*, *contingency* and
   *overload energy* ([Chapter 19](../../part-3-power-systems/ch19-operational-security/index.md)), *busbar*,
   *coupler* and *relevant substation* ([Chapter 13](../../part-3-power-systems/ch13-substations-grid-models/index.md)),
   *slack*, *distributed slack* and *PST* ([Chapter 15](../../part-3-power-systems/ch15-ac-power-flow/index.md),
   [Chapter 18](../../part-3-power-systems/ch18-sensitivity-factors-2/index.md)), *action set*
   ([Chapter 22](../../part-3-power-systems/ch22-import-preprocessing/index.md)), *static information*, `jit`
   and *XLA* ([Chapter 31](../../part-5-computing/ch31-jax-gpu-fundamentals/index.md)).

## Exercise 1

Electricity does not take the route an operator chooses: on a meshed grid it divides itself among all the
parallel paths between where it is produced and where it is used, and the share of each path is set by the
physics of the wires (a shorter, thicker wire takes more). When one line is overloaded, the operator can pay
power plants to produce elsewhere, so that less power needs that corridor, which costs money. Or the operator can
take one line *out*: the power that used that line now spreads over the remaining paths, and if the line removed
was feeding into the overloaded corridor, the overloaded line ends up carrying *less*. Removing a wire can also
make a *different* line worse, so the operator checks every line before and after. [Chapter 16](../../part-3-power-systems/ch16-dc-power-flow/index.md)
turns "the physics of the wires" into a linear model where this effect is a single number per line pair, and
[Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md) names it the LODF.

## Exercise 2

| Action | What changes physically | Cost to execute |
|---|---|---|
| Line switching | A circuit breaker opens (or closes) at both ends of a line; the line carries no current and the rest of the grid takes its flow | No fuel or market payment; wear on the breaker and a lower security margin (one path fewer) |
| Busbar splitting | In a substation with two busbars, the coupler between them opens; the lines, generators and loads of that substation are now in two electrical nodes, each with the branches connected to its busbar | No payment; a few switching operations, done remotely |
| Busbar reassignment | A branch's disconnectors move it from busbar A to busbar B (or back) while the coupler stays open or closed as decided; the branch joins the other node | No payment; more switching operations than a plain split, and a short interruption of the branch if it cannot be transferred live |
| PST tap change | The tap changer of a phase-shifting transformer moves one or more steps, changing the phase angle the transformer adds; flow is pushed from one corridor to another | No payment; mechanical wear of the tap changer, and the operating range is limited by the tap positions |

The point of the chapter: all four are *non-costly* in the sense of the market (no redispatch payment), which is
why ToOp searches among them before an operator pays for redispatch. None is free of operational consequences
(switching wear, protection settings, reduced margins), which is why the AC validation stage exists.

## Exercise 3

With ToOp's environment, both `JAX_PLATFORMS=cpu python -c "import jax; print(jax.devices())"` and the same
line without the variable printed `[CpuDevice(id=0)]` on this machine. The output shows that JAX will put every
array and every computation on the CPU, whatever the hardware: `JAX_PLATFORMS` is a *restriction* of the backends
JAX may use, not a request for one. On a machine where the `toop-gpu` service works, the same command without the
variable prints `[CudaDevice(id=0)]`; with it, `[CpuDevice(id=0)]`. The variable therefore lets you run the
*same* code and the *same* container image on both devices and compare: the wall-clock time of the solver (the
second run of a session, see Lab step 2), and the results, which should agree to floating-point precision since
ToOp forces `float64` on both. It is also the first thing to check when a GPU container "runs slowly": a stray
`JAX_PLATFORMS=cpu` in the environment is one of the silent failures the device banner names.

## Exercise 4

Run with ToOp's interpreter at commit `ba1874e`, the command `pytest packages/interfaces_pkg/tests -q` (with
ToOp's own `addopts` disabled, which only changes the report format) printed `160 passed, 4 warnings in 101.94s`,
1 min 43 s of wall-clock time. The exact count changes with the ToOp commit, and the time with the machine;
keep yours. The four warnings come from libraries (pandera, pydantic) and from one test using a `Path` as a
context manager, not from the interfaces package itself. [Chapter 30](../../part-5-computing/ch30-scientific-python/index.md)
explains what such a suite guarantees and [Chapter 33](../../part-5-computing/ch33-testing-debugging-profiling/index.md)
how to read it.

## Exercise 5

`n_nodes` is 72 and `n_branches` is 82 (`counts(si)` in `case57_toop.py`; the notebook's own Markdown states the
same two numbers). The hypothesis to form: *the solver does not count buses, it counts electrical nodes, and a
substation can be more than one node*. The breakdown from `node_type_counts` and `branch_type_counts`:

| Type | Count | Where it comes from |
|---|---|---|
| `BUS` | 58 | the 57 IEEE buses plus `PSTBus`, the extra bus that `powsybl_extended_case57` adds to hang a phase-shifting transformer on |
| `PSTNode` | 1 | the internal node of that PST, added at the stage `combine_phaseshift_and_injection` (`n_nodes` 58 to 59) so that the DC solver can control its angle |
| `REDUCED_NODE` | 1 | a node the preprocessing adds while it handles bridges (the log shows `n_nodes` go from 59 to 60 at the stage `exclude_bridges_from_outage_masks`) |
| `BUS_B` | 12 | one second busbar for each of the 12 substations that keep at least one split action (`relevant substations [4, 7, 12, 16, 17, 19, 20, 21, 24, 33, 46, 50]`, zero-based node indices); the preprocessing log shows `n_nodes` jump from 60 to 72 at the stage `add_bus_b_columns_to_ptdf` |

So $72 = 57 + 1 + 1 + 1 + 12$. The branches are the 80 of IEEE 57 (63 lines and 17 transformers) plus the two
elements of the fake PST, `PSTLine` and `PST`: 64 lines and 18 two-winding transformers, $82 = 80 + 2$.
`create_busbar_b_in_ieee` in the generator gives *every* bus a busbar B and a coupler, but only the substations
that can be split (four or more branches, [Chapter 19](../../part-3-power-systems/ch19-operational-security/index.md),
Exercise 8) become two solver nodes; the others collapse back into one. [Chapter 13](../../part-3-power-systems/ch13-substations-grid-models/index.md)
explains the busbar model and [Chapter 22](../../part-3-power-systems/ch22-import-preprocessing/index.md) the
preprocessing stages that produce these counts.
