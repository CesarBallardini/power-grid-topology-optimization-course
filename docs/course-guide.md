# Course guide

## Audience and prerequisites

Students entering the course have:

- College algebra and analytic geometry (vectors, systems of equations, basic matrices).
- Single- and multivariable calculus (derivatives, partial derivatives, Taylor series, integrals).
- Sophomore physics, including an introduction to electricity and magnetism.
- An introductory programming course at most. [Chapter 30](part-5-computing/ch30-scientific-python.md)
  (Scientific Python) runs in parallel from the first semester.

The course takes **five semesters** at roughly **8–10 hours of study per week**, alongside other courses
(80.5 weeks in total). Weeks per chapter are estimates for planning, not deadlines.

## Structure

| Part | Chapters | Weeks |
|---|---|---|
| Part 0 — Orientation | [0 Orientation and setup](part-0-orientation/ch00-orientation-and-setup.md) | 1 |
| Part I — Mathematical foundations | [1 Complex numbers and phasor arithmetic](part-1-math/ch01-complex-numbers.md) · [2 Linear algebra for networks](part-1-math/ch02-linear-algebra.md) · [3 Graph theory for power grids](part-1-math/ch03-graph-theory.md) · [4 Numerical linear algebra and floating point](part-1-math/ch04-numerical-linear-algebra.md) · [5 Probability for stochastic search](part-1-math/ch05-probability.md) · [6 Nonlinear equations and Newton–Raphson](part-1-math/ch06-newton-raphson.md) | 12.5 |
| Part II — Electricity and circuits | [7 Electromagnetism refresher](part-2-circuits/ch07-electromagnetism.md) · [8 DC circuits and nodal analysis in matrix form](part-2-circuits/ch08-dc-circuits.md) · [9 AC steady state and complex power](part-2-circuits/ch09-ac-steady-state.md) · [10 Three-phase systems and the per-unit system](part-2-circuits/ch10-three-phase-per-unit.md) · [11 Transformers, phase-shifting transformers, machines and HVDC](part-2-circuits/ch11-transformers-psts-hvdc.md) | 11 |
| Part III — Power systems and grid computations | [12 The power grid and its operation](part-3-power-systems/ch12-power-grid-operation.md) · [13 Substations, grid models and data formats](part-3-power-systems/ch13-substations-grid-models.md) · [14 Network component models and network matrices](part-3-power-systems/ch14-component-models.md) · [15 AC power flow](part-3-power-systems/ch15-ac-power-flow.md) · [16 DC power flow](part-3-power-systems/ch16-dc-power-flow.md) · [17 Sensitivity factors I: PTDF, PSDF, LODF and N-1 screening](part-3-power-systems/ch17-sensitivity-factors-1.md) · [18 Sensitivity factors II: low-rank updates for topology changes](part-3-power-systems/ch18-sensitivity-factors-2.md) · [19 Operational security and performance metrics](part-3-power-systems/ch19-operational-security.md) · [20 Protection, short circuit and switching feasibility](part-3-power-systems/ch20-protection-short-circuit.md) · [21 Coordinated security analysis, capacity calculation and remedial action optimization](part-3-power-systems/ch21-coordinated-security-rao.md) · [22 Import and preprocessing: from grid model to search space](part-3-power-systems/ch22-import-preprocessing.md) | 24.5 |
| Part IV — Optimization | [23 Optimization modeling: LP, duality, MILP and complexity](part-4-optimization/ch23-optimization-modeling.md) · [24 Heuristics and evolutionary algorithms](part-4-optimization/ch24-evolutionary-algorithms.md) · [25 Quality-diversity and MAP-Elites](part-4-optimization/ch25-quality-diversity.md) · [26 Experimental methodology for stochastic optimizers](part-4-optimization/ch26-experimental-methodology.md) · [27 True Pareto multi-objective optimization](part-4-optimization/ch27-pareto-multi-objective.md) · [28 Topology optimization in power systems](part-4-optimization/ch28-topology-optimization.md) · [29 Learning-based topology control](part-4-optimization/ch29-learning-based-control.md) | 15.5 |
| Part V — Computing | [30 Scientific Python](part-5-computing/ch30-scientific-python.md) · [31 JAX and GPU computing fundamentals](part-5-computing/ch31-jax-gpu-fundamentals.md) · [32 JAX in ToOp: internals and GPU tuning](part-5-computing/ch32-jax-in-toop.md) · [33 Testing, debugging and profiling scientific and JAX code](part-5-computing/ch33-testing-debugging-profiling.md) · [34 Engineering the pipeline](part-5-computing/ch34-pipeline-engineering.md) | 11 |
| Part VI — Capstone | [35 Capstone: reading ToOp end to end](part-6-capstone/ch35-capstone.md) | 5 |
| **Total** | 36 chapters | **80.5** |

## Five-semester schedule

Chapters are numbered by part; the teaching order interleaves the parts so that every prerequisite, including
the prerequisites of each lab, is taught first. The dependency flowchart is in the
[introduction](index.md#chapters-and-their-dependencies).

| Semester | Chapters, in teaching order | Weeks |
|---|---|---|
| 1 | [0](part-0-orientation/ch00-orientation-and-setup.md), [1](part-1-math/ch01-complex-numbers.md), [2](part-1-math/ch02-linear-algebra.md), [3](part-1-math/ch03-graph-theory.md), [4](part-1-math/ch04-numerical-linear-algebra.md), [5](part-1-math/ch05-probability.md), [30](part-5-computing/ch30-scientific-python.md) | 15.5 |
| 2 | [6](part-1-math/ch06-newton-raphson.md), [7](part-2-circuits/ch07-electromagnetism.md), [8](part-2-circuits/ch08-dc-circuits.md), [9](part-2-circuits/ch09-ac-steady-state.md), [10](part-2-circuits/ch10-three-phase-per-unit.md), [11](part-2-circuits/ch11-transformers-psts-hvdc.md), [12](part-3-power-systems/ch12-power-grid-operation.md), [13](part-3-power-systems/ch13-substations-grid-models.md) | 16 |
| 3 | [14](part-3-power-systems/ch14-component-models.md), [15](part-3-power-systems/ch15-ac-power-flow.md), [16](part-3-power-systems/ch16-dc-power-flow.md), [17](part-3-power-systems/ch17-sensitivity-factors-1.md), [31](part-5-computing/ch31-jax-gpu-fundamentals.md), [18](part-3-power-systems/ch18-sensitivity-factors-2.md), [23](part-4-optimization/ch23-optimization-modeling.md) | 16.5 |
| 4 | [19](part-3-power-systems/ch19-operational-security.md), [20](part-3-power-systems/ch20-protection-short-circuit.md), [21](part-3-power-systems/ch21-coordinated-security-rao.md), [22](part-3-power-systems/ch22-import-preprocessing.md), [24](part-4-optimization/ch24-evolutionary-algorithms.md), [25](part-4-optimization/ch25-quality-diversity.md), [32](part-5-computing/ch32-jax-in-toop.md), [33](part-5-computing/ch33-testing-debugging-profiling.md) | 16 |
| 5 | [26](part-4-optimization/ch26-experimental-methodology.md), [27](part-4-optimization/ch27-pareto-multi-objective.md), [28](part-4-optimization/ch28-topology-optimization.md), [29](part-4-optimization/ch29-learning-based-control.md), [34](part-5-computing/ch34-pipeline-engineering.md), [35](part-6-capstone/ch35-capstone.md) | 16.5 |

### End-of-semester checkpoints

Each semester ends with a practical checkpoint that later semesters build on:

| Semester | Checkpoint |
|---|---|
| 1 | A nodal network solver in NumPy; bridges and articulation points of a grid graph; a Sherman–Morrison update verified against a re-solve |
| 2 | A per-unit two-bus model with a phase-shifting transformer; a Newton–Raphson solver; reading a substation's node-breaker model |
| 3 | PTDF, LODF and BSDF implemented from scratch, validated against ToOp's `numpy_reference.py`, and ported to JAX; an LP/MILP model |
| 4 | ToOp run end to end on the node-breaker example grid, with metrics recomputed by hand and a seeded bug found and fixed |
| 5 | Capstone project with a reproducible experiment and an operator brief |

## Lab tools

Every lab states the tools it uses. Students need:

| Tool | Used for | Link |
|---|---|---|
| Jupyter notebooks | Most labs and exercises | <https://jupyter.org> |
| NumPy, pandas, matplotlib | Matrices, tables and plots | <https://numpy.org>, <https://pandas.pydata.org>, <https://matplotlib.org> |
| GNU Octave | Linear algebra, numerical methods, and MATPOWER power flow cases | <https://octave.org> |
| MATPOWER | Power flow, DC power flow and sensitivity factors in Octave | <https://matpower.org> |
| GeoGebra | Phasors, the complex plane, two-bus power transfer, LP geometry, Pareto fronts | <https://www.geogebra.org> |
| pandapower and pypowsybl | Grid models, AC/DC power flow, contingency analysis | <https://pandapower.readthedocs.io/>, <https://powsybl.readthedocs.io/projects/pypowsybl/en/latest/> |
| ToOp | The engine studied in the course, run in a container | <https://github.com/eliagroup/ToOp> |

[Chapter 0](part-0-orientation/ch00-orientation-and-setup.md) sets up ToOp in a container (the supported
path on Windows). A GPU is optional: every lab runs on CPU, with smaller grids.

## Printable version

The whole book is also available as a single page, with a table of contents, for printing or saving as PDF
from the browser: [print version](../print_page/).

## Conventions

### Access labels

Book access was checked against the archive.org metadata and loans APIs in September 2026.

| Label | Meaning |
|---|---|
| **FREE** | Openly licensed or legitimately hosted by the author, publisher or university |
| **BORROW** | archive.org lending library (`inlibrary` collection): anyone with a free account can borrow it |
| **PRINT-DISABLED** | archive.org copy restricted to users certified as print-disabled; not generally available |
| **NOT ON IA** | No library copy found on archive.org; use a university library |
| **USER-UPLOAD (ES)** | A Spanish scan uploaded by an archive.org user, listed only for books with no FREE or BORROW copy in any language |

!!! warning "About USER-UPLOAD (ES) links"

    These items are not part of archive.org's lending library. They are scans uploaded by individual users,
    their copyright status is unclear, and archive.org may remove them at any time. They are listed only as a
    last resort for books that cannot otherwise be read online; when a university library copy is reachable,
    use it instead.

Borrowable copies are often older editions. Core material in these subjects changes little between
editions, but **chapter and section numbers vary by edition**: check the table of contents of the copy you
have.

### Spanish editions

All course material is in English. When a textbook has a Spanish edition, the book entry adds
`ES: *Título* (publisher, edition, year)`. Two recommended books are Spanish originals: Gómez-Expósito
(coord.), *Análisis y operación de sistemas de energía eléctrica*, and Fraile Mora, *Máquinas eléctricas*.
[Appendix B](appendices/b-books.md) lists all Spanish editions found.

### Syllabus counts

Counts such as "(12 syllabi)" give the number of distinct universities whose course bibliography cites the
book, from the survey in [Appendix A](appendices/a-syllabi.md).

### ToOp source paths

ToOp source paths are abbreviated:

| Prefix | Full path in the ToOp repository |
|---|---|
| `dc_solver/` | `packages/dc_solver_pkg/src/toop_engine_dc_solver/` |
| `optimizer/` | `packages/topology_optimizer_pkg/src/toop_engine_topology_optimizer/` |
| `interfaces/` | `packages/interfaces_pkg/src/toop_engine_interfaces/` |
| `grid_helpers/` | `packages/grid_helpers_pkg/src/toop_engine_grid_helpers/` |
| `contingency/` | `packages/contingency_analysis_pkg/src/toop_engine_contingency_analysis/` |
| `importer/` | `packages/importer_pkg/src/toop_engine_importer/` |
| `docs/`, `notebooks/`, `packages/` | Same name at the repository root |

Paths were checked against ToOp commit `ba1874e` (September 2026).

### Where cited ToOp files come from

Not every ToOp file cited in the book is part of the upstream repository:

| Files | Where they are |
|---|---|
| `packages/`, `docs/`, `notebooks/`, `data/`, `toop-engine-benchmark/` | Upstream [eliagroup/ToOp](https://github.com/eliagroup/ToOp), commit `ba1874e` |
| `docker-compose.yaml`, `docker/Dockerfile`, `docker/README.md`, `docker/device_banner.py` | Branch `feat/add-docker-compose` of the course fork [CesarBallardini/ToOp](https://github.com/CesarBallardini/ToOp) (container setup used in [Chapter 0](part-0-orientation/ch00-orientation-and-setup.md)) |
