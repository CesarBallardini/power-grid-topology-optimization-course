# Power Grid Topology Optimization

*From college algebra to reading, running, debugging and improving ToOp.*

## Introduction

This course has one goal: that a student who starts with sophomore algebra, calculus and physics can
**comprehend, use, debug and improve** the optimization code in [ToOp](https://github.com/eliagroup/ToOp),
the open-source GPU topology optimization engine by Elia Group and 50Hertz, and topology optimization software
in general. ToOp answers a question grid operators face every day: *which switching actions inside the
transmission grid remove overloads, both in normal operation and after any single outage, without paying
generators to change their output?*

Answering that question in software takes several bodies of knowledge, and the book is organized around them:

1. **Mathematics (Part I).** The DC load flow solver is applied linear algebra on a graph: a weighted
   Laplacian, a sparse solve, and low-rank matrix updates. AC validation needs complex numbers and Newton's
   method; evolutionary search needs basic probability.
2. **Electricity and circuits (Part II).** Grids are circuits. Nodal analysis, phasors, complex power,
   three-phase systems, per-unit values and transformer models turn physical equipment into equations.
3. **Power systems (Part III).** Substation topology and grid data, AC power flow and its DC approximation,
   the sensitivity factors that make ToOp fast (PTDF, LODF, MODF, BSDF), the N-1 security rules and metrics
   that define an acceptable grid state, short-circuit and protection limits on what can be switched, the
   European processes in which remedial actions are coordinated, and the preprocessing that turns a raw grid
   model into a search space. Chapters 17 and 18 are the mathematical core of the course.
4. **Optimization (Part IV).** Why the exact switching problem is hard, how evolutionary and
   quality-diversity search (MAP-Elites) work, how to run fair experiments with stochastic optimizers, how
   true Pareto multi-objective methods differ from the weighted sum ToOp uses, where ToOp sits among exact,
   heuristic and learning-based topology optimization methods.
5. **Computing (Part V).** A parallel track from scientific Python in the first semester to JAX on GPUs,
   ToOp's JAX internals, testing, debugging and profiling numerical code, and the engineering of a
   message-driven pipeline.

Chapter 0 opens the course by setting up ToOp and running it once as a black box, so that every later
chapter answers a question students have already asked. The capstone (Part VI) brings everything together on
the real code base, with a written brief for grid operators.

Labs and exercises use Jupyter notebooks with NumPy, pandas and matplotlib, Octave (with MATPOWER for power
flow), GeoGebra for geometric intuition, pandapower and pypowsybl, and ToOp itself. The
[course guide](course-guide.md) explains the schedule, tools and conventions.

## Chapters and their dependencies

Each arrow points from a prerequisite to the chapter that needs it, including the prerequisites of each
chapter's lab. To keep the chart readable, an arrow is omitted when the dependency already follows from a longer
path (2 → 8 is implied by 2 → 3 → 8); the "Prerequisites" line at the top of every chapter lists them all. The
same data drives the five-semester schedule in the [course guide](course-guide.md#five-semester-schedule). Node colors mark the parts: gray for
orientation, blue for mathematics, yellow for electricity and circuits, orange for power systems, green for
optimization, purple for computing and pink for the capstone.

```mermaid
flowchart TD
    C0["0 Orientation"]:::p0
    C1["1 Complex numbers"]:::p1
    C2["2 Linear algebra"]:::p1
    C3["3 Graph theory"]:::p1
    C4["4 Numerical linear algebra"]:::p1
    C5["5 Probability"]:::p1
    C6["6 Newton–Raphson"]:::p1
    C7["7 Electromagnetism"]:::p2
    C8["8 DC circuits"]:::p2
    C9["9 AC steady state"]:::p2
    C10["10 Three-phase, per unit"]:::p2
    C11["11 Transformers, PSTs"]:::p2
    C12["12 Grid operation"]:::p3
    C13["13 Substations, grid models"]:::p3
    C14["14 Component models"]:::p3
    C15["15 AC power flow"]:::p3
    C16["16 DC power flow"]:::p3
    C17["17 PTDF, LODF"]:::p3
    C18["18 MODF, BSDF"]:::p3
    C19["19 Security, metrics"]:::p3
    C20["20 Protection, short circuit"]:::p3
    C21["21 Coordinated RAO"]:::p3
    C22["22 Import, preprocessing"]:::p3
    C23["23 LP, MILP"]:::p4
    C24["24 Evolutionary algorithms"]:::p4
    C25["25 MAP-Elites"]:::p4
    C26["26 Experimental methodology"]:::p4
    C27["27 Pareto optimization"]:::p4
    C28["28 Topology optimization"]:::p4
    C29["29 Learning-based control"]:::p4
    C30["30 Scientific Python"]:::p5
    C31["31 JAX, GPU"]:::p5
    C32["32 JAX in ToOp"]:::p5
    C33["33 Testing, debugging"]:::p5
    C34["34 Pipeline engineering"]:::p5
    C35["35 Capstone"]:::p6
    C2 --> C3
    C2 --> C4
    C4 --> C6
    C3 --> C8
    C7 --> C8
    C1 --> C9
    C8 --> C9
    C9 --> C10
    C10 --> C11
    C9 --> C12
    C12 --> C13
    C11 --> C14
    C12 --> C14
    C6 --> C15
    C14 --> C15
    C15 --> C16
    C16 --> C17
    C13 --> C18
    C17 --> C18
    C17 --> C19
    C18 --> C20
    C19 --> C20
    C19 --> C21
    C18 --> C22
    C19 --> C22
    C2 --> C23
    C5 --> C24
    C17 --> C24
    C23 --> C24
    C24 --> C25
    C30 --> C25
    C25 --> C26
    C26 --> C27
    C19 --> C28
    C27 --> C28
    C28 --> C29
    C31 --> C29
    C30 --> C31
    C18 --> C32
    C31 --> C32
    C32 --> C33
    C25 --> C34
    C22 --> C35
    C29 --> C35
    C33 --> C35
    C34 --> C35
    classDef p0 fill:#eeeeee,stroke:#555,color:#111
    classDef p1 fill:#dbe9ff,stroke:#555,color:#111
    classDef p2 fill:#fff1c9,stroke:#555,color:#111
    classDef p3 fill:#ffd9cc,stroke:#555,color:#111
    classDef p4 fill:#dcf5dc,stroke:#555,color:#111
    classDef p5 fill:#eadcf8,stroke:#555,color:#111
    classDef p6 fill:#f8d7e8,stroke:#555,color:#111
```

The longest prerequisite chain runs **2 → 3 → 8 → 9 → 10 → 11 → 14 → 15 → 16 → 17 → 24 → 25 → 26 → 27 → 28 →
29 → 35**: linear algebra and graphs lead through circuits, component models and power flow to the
sensitivity factors, which become the fitness evaluator of the evolutionary and quality-diversity search, and
from there to multi-objective and learning-based topology optimization and the capstone.

## How each chapter is organized

Every chapter page has the same structure:

- **Weeks, semester and prerequisites**, with links.
- **Why it matters for ToOp**: the part of the code or documentation that depends on the chapter.
- **Topics** and **learning outcomes**.
- **Resources**: courses and videos, free texts, textbooks (with archive.org access status and Spanish
  editions), papers and documentation.
- **Lab**: a hands-on session, most of them on real ToOp code or data.
- **Exercises**: short pen-and-paper and computational problems.

Access labels, syllabus counts and the abbreviations used for ToOp source paths are defined in the
[conventions](course-guide.md#conventions).

The appendices record the evidence behind the plan: the [university syllabi](appendices/a-syllabi.md)
consulted, the [bibliography](appendices/b-books.md) with access status and Spanish editions, and a
[concept map of the ToOp code](appendices/c-toop-concept-map.md). The [glossary](appendices/d-glossary.md)
defines the acronyms and technical terms used throughout; hovering over an acronym anywhere in the book shows
its expansion.
