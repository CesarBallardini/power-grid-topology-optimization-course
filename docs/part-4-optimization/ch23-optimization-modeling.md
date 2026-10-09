# Chapter 23. Optimization modeling: LP, duality, MILP and complexity

**Weeks:** 2 · **Semester:** 3 · **Prerequisites:** [Chapter 2](../part-1-math/ch02-linear-algebra.md)

!!! info "Why it matters for ToOp"

    Choosing which substations to split and which lines to switch is a combinatorial decision. Written exactly, it
    is a mixed-integer program, and the problem family is NP-hard. This chapter gives the generic vocabulary you
    need to read any topology optimization paper and to judge ToOp's design choice:

    - **Search-space size.** ToOp's exhaustive baseline counts its candidates as a sum of products over
      substation action sets times a sum of binomial coefficients over disconnectable branches
      (`count_workset_size` in `optimizer/dc_bruteforce/generator.py`). Estimating that number is an exercise
      in this chapter.
    - **Why not a MILP solver.** Branch and bound gives optimality certificates but its running time grows
      exponentially in the worst case; ToOp instead evaluates huge batches of candidates on a GPU with a heuristic
      search ([Chapter 24](ch24-evolutionary-algorithms.md), [Chapter 25](ch25-quality-diversity.md)).
    - **Dual values and relaxations** reappear in optimal power flow, nodal prices and convex relaxations.

    The power-system formulations (DC-OPF, optimal transmission switching as a MILP and its complexity) are in
    [Chapter 28](ch28-topology-optimization.md), after you have seen the heuristics. This chapter stays generic.

## Topics

- Modeling: decision variables, objective, constraints; standard and canonical forms; modeling languages
  (PuLP, Pyomo) and solvers (HiGHS, GLPK).
- Linear programming: geometry of polyhedra, vertices and basic feasible solutions, the simplex method,
  degeneracy and unboundedness.
- Duality: weak and strong duality, complementary slackness, shadow prices, sensitivity (ranging) analysis.
- Convex sets and functions; convex vs non-convex problems; why local optima of convex problems are global.
- Integer and mixed-integer linear programming: logical constraints with binaries, big-M formulations and why
  a large $M$ weakens the LP relaxation; branch and bound; cutting planes; optimality gap.
- Computational complexity: P, NP, NP-completeness and NP-hardness, reductions (knapsack, set cover); exponential
  search spaces and what "hard in the worst case" does and does not mean in practice.

## Learning outcomes

- Formulate a small allocation or location problem as an LP or MILP, solve it with HiGHS (or Octave `glpk`),
  and interpret the dual values of the binding constraints.
- Solve a two-variable LP graphically and explain why an optimum sits at a vertex.
- Write a big-M formulation of an either-or constraint, compare its LP relaxation bound with the MILP optimum
  for two values of $M$, and explain the effect on branch and bound.
- Estimate the size of a combinatorial search space (for example, ToOp's split and switching candidates) and
  argue from NP-hardness why exact enumeration does not scale.

## Resources

- Courses: MIT OCW 6.251J *Introduction to Mathematical Programming*
  <https://ocw.mit.edu/courses/6-251j-introduction-to-mathematical-programming-fall-2009/>; MIT OCW 15.053
  <https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/>; Stanford
  EE364a / edX CVX101 *Convex Optimization* <https://web.stanford.edu/class/ee364a/>; Coursera *Basic Modeling
  for Discrete Optimization* (Melbourne) <https://www.coursera.org/learn/basic-modeling>.
- Free texts:
  - Boyd & Vandenberghe, *Convex Optimization* (6 syllabi), Ch. 2–5: FREE
    <https://web.stanford.edu/~boyd/cvxbook/>.
  - Postek, Zocca, Gromicho & Kantor, *Hands-On Mathematical Optimization with Python* (Pyomo): FREE
    <https://mobook.github.io/MO-book/>.
  - Bradley, Hax & Magnanti, *Applied Mathematical Programming*: FREE <https://web.mit.edu/15.053/www/>.
- Textbooks:
  - Bertsimas & Tsitsiklis, *Introduction to Linear Optimization* (4 syllabi; required in US graduate
    courses): Ch. 1–4 (formulation, simplex, duality), Ch. 10–11 (integer programming). **BORROW**
    [`introductiontoli0000bert`](https://archive.org/details/introductiontoli0000bert).
  - Hillier & Lieberman, *Introduction to Operations Research* (6 syllabi; the Latin American favorite). 7th
    ed. 2000 **BORROW** [`introductiontoop0000fred_7edi`](https://archive.org/details/introductiontoop0000fred_7edi).
    ES: *Introducción a la investigación de operaciones* (McGraw-Hill).
  - Winston, *Operations Research: Applications and Algorithms* (7 syllabi, the most cited). **PRINT-DISABLED**.
    ES: *Investigación de operaciones: aplicaciones y algoritmos* (Thomson, 4th ed. 2005).
  - Taha, *Operations Research: An Introduction* (3 syllabi). **PRINT-DISABLED**.
    ES: *Investigación de operaciones* (Pearson, 9th ed.), **USER-UPLOAD (ES)** [`investigacion-de-operaciones-9a-ed-taha`](https://archive.org/details/investigacion-de-operaciones-9a-ed-taha).
  - Chvátal, *Linear Programming* (1983). **BORROW** [`linearprogrammin00chv`](https://archive.org/details/linearprogrammin00chv).
  - Papadimitriou & Steiglitz, *Combinatorial Optimization* (1982): simplex and duality, integer programming,
    NP-completeness, branch and bound, local search. **BORROW**
    [`combinatorialopt0000papa`](https://archive.org/details/combinatorialopt0000papa).
  - Garey & Johnson, *Computers and Intractability: A Guide to the Theory of NP-Completeness* (1979), Ch. 1–3
    and the knapsack and set-cover entries of the problem list. **PRINT-DISABLED**.
- Documentation:
  - PuLP <https://coin-or.github.io/pulp/>; Pyomo <https://www.pyomo.org/>; HiGHS <https://highs.dev/>.
  - GNU Octave, "Linear Programming" (`glpk`) <https://docs.octave.org/latest/Linear-Programming.html>.
  - GeoGebra Graphing Calculator <https://www.geogebra.org/calculator>.

## Worked example

A tiny uncapacitated facility location problem shows every modeling idea of the chapter. Open facility
$i \in I$ at fixed cost $f_i$ ($y_i \in \{0,1\}$) and serve customer $j \in J$ from $i$ at cost $c_{ij}$
($x_{ij} \ge 0$):

$$
\min_{x,\,y} \; \sum_{i \in I} f_i y_i + \sum_{i \in I}\sum_{j \in J} c_{ij} x_{ij}
\quad \text{s.t.} \quad \sum_{i \in I} x_{ij} = 1 \;\; \forall j, \qquad
x_{ij} \le y_i \;\; \forall i,j, \qquad y_i \in \{0,1\}.
$$

The linking constraints $x_{ij} \le y_i$ are the "strong" formulation. The aggregated version
$\sum_j x_{ij} \le |J|\, y_i$ is a big-M constraint with $M = |J|$: it describes the same integer solutions but a
weaker LP relaxation, so branch and bound explores more nodes.

```python
import pulp

f = {"A": 10, "B": 12, "C": 8}
c = {("A", 1): 2, ("A", 2): 4, ("A", 3): 5, ("A", 4): 7,
     ("B", 1): 6, ("B", 2): 3, ("B", 3): 2, ("B", 4): 4,
     ("C", 1): 7, ("C", 2): 6, ("C", 3): 5, ("C", 4): 2}
I, J = list(f), [1, 2, 3, 4]

def solve(strong: bool, relax: bool):
    m = pulp.LpProblem("ufl", pulp.LpMinimize)
    y = pulp.LpVariable.dicts("open", I, 0, 1, cat="Continuous" if relax else "Binary")
    x = pulp.LpVariable.dicts("assign", c, lowBound=0)
    m += pulp.lpSum(f[i] * y[i] for i in I) + pulp.lpSum(c[k] * x[k] for k in c)
    for j in J:
        m += pulp.lpSum(x[i, j] for i in I) == 1
    if strong:
        for i, j in c:
            m += x[i, j] <= y[i]
    else:  # big-M with M = |J|
        for i in I:
            m += pulp.lpSum(x[i, j] for j in J) <= len(J) * y[i]
    m.solve(pulp.HiGHS(msg=False))
    return pulp.value(m.objective), {i: y[i].value() for i in I}

for strong in (True, False):
    print("strong" if strong else "big-M", solve(strong, relax=False), solve(strong, relax=True))
```

Both formulations find the optimum 27 (open only B). The LP relaxation of the strong formulation is already
27, but the big-M relaxation drops to 19.5 with fractional $y = (0.25, 0.5, 0.25)$: that gap is the work branch
and bound has to close, and on larger instances it is the difference between seconds and hours.

## Lab

!!! example "Lab: LP geometry, duality and big-M in practice"

    **Tools:** GeoGebra (part 1); Jupyter with PuLP + HiGHS, NumPy, pandas and matplotlib (parts 2–3); GNU Octave
    `glpk` is an accepted alternative for part 2.

    1. **2-D LP in GeoGebra.** Draw the feasible region of
       $\max\, 3x_1 + 5x_2$ s.t. $x_1 \le 4$, $2x_2 \le 12$, $3x_1 + 2x_2 \le 18$, $x \ge 0$. Add a slider for the
       objective level $z$ and the line $3x_1 + 5x_2 = z$; find the last vertex it touches. Add a slider for the
       right-hand side $18$ and read the change of the optimum per unit: that is the shadow price. Rotate the
       objective with a second slider until the optimum jumps to a neighboring vertex (ranging).
    2. **Duality check.** Solve the same LP in PuLP (or Octave `glpk`), print the dual values
       (`constraint.pi` in PuLP; `extra.lambda` from `[x, fmin, status, extra] = glpk(...)` in Octave; mind each
       tool's sign convention for maximization problems) and verify complementary slackness and strong
       duality by hand.
    3. **MILP and big-M.** Generate random uncapacitated facility location instances with NumPy (10–40 sites,
       30–120 customers). Solve the strong formulation and the big-M formulation from the worked example; record
       in a pandas table the LP relaxation value, the MILP optimum, the solve time and, if the HiGHS log shows
       it, the number of branch-and-bound nodes. Plot time against instance size with matplotlib on a log scale
       and write three sentences on what the plot says about exact methods for combinatorial problems.

## Exercises

1. Convert $\max\, 2x_1 - x_2$ s.t. $x_1 + x_2 \le 4$, $x_1 - x_2 \le 1$, $x_1 \ge 0$, $x_2$ free, into
   standard form and write its dual. Solve both graphically and check that the optimal values agree.
   *(GeoGebra)*
2. Model "either $x_1 + x_2 \le 4$ or $3x_1 + x_2 \le 6$, with $0 \le x \le 10$" with one binary variable and a
   big-M. Derive the smallest valid $M$ for each constraint and show with a plot of the LP relaxation why a
   larger $M$ is worse. *(GeoGebra)*
3. Run branch and bound by hand on the knapsack problem $\max\, 5a + 4b + 3c$ s.t. $2a + 3b + c \le 5$,
   $a, b, c \in \{0,1\}$: draw the tree, the LP bound at each node and the pruning reason. Check with PuLP.
   *(Jupyter: PuLP + HiGHS)*
4. A grid has 40 relevant substations whose numbers of valid split actions you draw with
   `np.random.default_rng(0).integers(3, 61, size=40)`. At most 2 substations may be split simultaneously and
   up to 3 of 200 disconnectable branches may be switched off. Compute the number of candidate topologies with
   `itertools` and `math.comb` in the same way as `count_workset_size` (`optimizer/dc_bruteforce/generator.py`),
   and the time to enumerate them at $10^5$ evaluations per second. Repeat for 4 simultaneous splits.
   *(Jupyter: NumPy)*
5. Show that the decision version of 0-1 knapsack is in NP by describing a certificate and its check. Then write
   the $O(nW)$ dynamic program and explain why it does not contradict NP-completeness (pseudo-polynomial
   time). *(pen and paper)*
