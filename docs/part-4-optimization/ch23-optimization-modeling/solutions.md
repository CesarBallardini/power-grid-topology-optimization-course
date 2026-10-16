# Chapter 23 – Solutions

Solutions of the Lab and the Exercises of [Chapter 23](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`lp_duality.py`](src/lp_duality.py) and [`lp_duality.m`](src/lp_duality.m)
(Lab steps 1–2 and Exercise 1, PuLP and Octave `glpk`), [`facility_location.py`](src/facility_location.py) and
[`big_m_study.py`](src/big_m_study.py) (Lab step 3), [`either_or.py`](src/either_or.py),
[`knapsack_bb.py`](src/knapsack_bb.py), [`search_space.py`](src/search_space.py) and
[`knapsack_dp.py`](src/knapsack_dp.py) (Exercises 2–5). Each module's `main()` prints every number quoted below;
the solver is HiGHS through PuLP 3.

## Lab

1. **2-D LP in GeoGebra** (the GeoGebra construction itself is not run here; `lp_duality.py` checks every number
   you read off the screen). The feasible region of $\max 3x_1 + 5x_2$ s.t. $x_1 \le 4$, $2x_2 \le 12$,
   $3x_1 + 2x_2 \le 18$, $x \ge 0$ is a pentagon with vertices (0, 0), (4, 0), (4, 3), (0, 6) and (2, 6), where
   the objective takes 0, 12, 27, 30 and 36 (`vertices()`). Sliding the level line $3x_1 + 5x_2 = z$ upward, the
   last vertex it touches is (2, 6) with $z = 36$: the optimum of an LP sits at a vertex because the objective is
   linear, so along any edge it is monotone and the best point is an end of the edge. The slider on the right-hand
   side 18 (`rhs_ranging()`): $z = 30, 33, 36, 39, 42$ for $b_3 = 12, 15, 18, 21, 24$, one unit of objective per
   unit of $b_3$, the shadow price $y_3 = 1$; it holds for $12 < b_3 < 24$. Below 12 the vertex moves to the
   $x_2$ axis and the price becomes 2.5 ($x_1 = 0$ and $2x_2 \le b_3$ decides, worth $5/2$ per unit); above 24 the
   constraints $2x_2 \le 12$ and $x_1 \le 4$ fix the optimum at (4, 6), $z = 42$, and the price is 0. At the two
   ends the vertex is degenerate (three constraints meet) and a solver may report either neighbouring price: HiGHS
   gives 1 at $b_3 = 12$ and 0 at $b_3 = 24$. Rotating the
   objective (`objective_ranging()`, $c_2 = 5$ fixed): the optimum stays at (2, 6) for $0 < c_1 \le 7.5$ and jumps
   to (4, 3) beyond; at $c_1 = 7.5$ the level line is parallel to the edge $3x_1 + 2x_2 = 18$ and every point of
   that edge is optimal. At $c_1 = 0$ the optimum is the whole edge from (0, 6) to (2, 6).
2. **Duality check.** `solve_lab_lp()` builds the LP in PuLP and reads `constraint.pi`. PuLP hands HiGHS a
   minimization of $-c^\top x$, so for a maximization the reported `pi` are the negatives of the textbook shadow
   prices; with the sign undone, $y = (0, 1.5, 1)$. Octave's `glpk` with `sense = -1` returns `extra.lambda`
   $= (0, 1.5, 1)$ directly (`lp_duality.m`, whose test blocks run under `tests/test_octave_sources.py`).
   Complementary slackness: slacks $(2, 0, 0)$ and $y_k \cdot \text{slack}_k = (0, 0, 0)$: constraint 1 has slack,
   so its price is zero; constraints 2 and 3 bind, so their prices may be positive. Strong duality:
   $b^\top y = 4 \cdot 0 + 12 \cdot 1.5 + 18 \cdot 1 = 36 = z$. By hand, the dual is $\min 4y_1 + 12y_2 + 18y_3$
   s.t. $y_1 + 3y_3 \ge 3$, $2y_2 + 2y_3 \ge 5$, $y \ge 0$; with $y_1 = 0$ (constraint 1 is slack) and both dual
   constraints tight ($x_1, x_2 > 0$) the two equations give $y_3 = 1$, $y_2 = 1.5$.
3. **MILP and big-M.** `big_m_study.study()` draws sites and customers in the unit square (assignment cost 10 times
   the distance, fixed costs between 3 and 8) and solves four models per instance:

    | Sites × customers | Formulation | LP relaxation | MILP | Gap | LP s | MILP s | B&B nodes |
    |---|---|---|---|---|---|---|---|
    | 10 × 30 | strong | 79.78 | 79.78 | 0.0 % | 0.007 | 0.017 | 1 |
    | 10 × 30 | big-M | 56.66 | 79.78 | 29.0 % | 0.003 | 0.034 | 1 |
    | 20 × 60 | strong | 123.36 | 123.36 | 0.0 % | 0.019 | 0.042 | 1 |
    | 20 × 60 | big-M | 79.02 | 123.36 | 35.9 % | 0.010 | 0.088 | 1 |
    | 30 × 90 | strong | 175.75 | 175.75 | 0.0 % | 0.050 | 0.101 | 1 |
    | 30 × 90 | big-M | 112.92 | 175.75 | 35.8 % | 0.022 | 0.267 | 1 |
    | 40 × 120 | strong | 203.10 | 203.10 | 0.0 % | 0.077 | 0.182 | 1 |
    | 40 × 120 | big-M | 120.10 | 203.10 | 40.9 % | 0.039 | 0.728 | 1 |

    Times are from one run on the maintainer's laptop and vary by a few tens of percent. Three sentences for the
    plot (time against size on a log scale, left to the notebook): the strong formulation's relaxation is
    integral on every instance, so HiGHS proves optimality at the root and the time grows like that of an LP;
    the big-M relaxation is 29–41 % below the optimum and the solver needs 2 to 4 times longer, the extra work
    being cuts and presolve in the root node (HiGHS reports a single branch-and-bound node here, so the
    exponential tree never materializes on instances this small); the lesson is that the formulation, not the
    solver, decides whether an exact method is cheap, and that on a problem with a weak relaxation and no
    strengthening available the tree grows exponentially with the number of binaries, which is the situation of
    optimal transmission switching ([Chapter 28](../ch28-topology-optimization/index.md)).

## Exercise 1

Standard form: replace the free variable by $x_2 = x_2^+ - x_2^-$ with $x_2^\pm \ge 0$ and add slacks
$s_1, s_2 \ge 0$:

$$
\max\ 2x_1 - x_2^+ + x_2^- \quad\text{s.t.}\quad
x_1 + x_2^+ - x_2^- + s_1 = 4,\qquad x_1 - x_2^+ + x_2^- + s_2 = 1,\qquad x_1, x_2^\pm, s_1, s_2 \ge 0 .
$$

Dual (one variable per primal constraint, $y_1, y_2 \ge 0$ because the constraints are $\le$ in a maximization;
one constraint per primal variable, an inequality for $x_1 \ge 0$ and an equation for the free $x_2$):

$$
\min\ 4y_1 + y_2 \quad\text{s.t.}\quad y_1 + y_2 \ge 2,\qquad y_1 - y_2 = -1,\qquad y \ge 0 .
$$

Graphically (GeoGebra, not run here; `exercise_1()` solves both with HiGHS): the primal region is the wedge
$x_1 - 1 \le x_2 \le 4 - x_1$, $x_1 \ge 0$, closed at $x_1 = 2.5$ where the two lines meet. The objective
$2x_1 - x_2$ prefers small $x_2$, so along the lower edge $x_2 = x_1 - 1$ it equals $x_1 + 1$, largest at the
corner (2.5, 1.5): $z^\ast = 5 - 1.5 = 3.5$. In the dual, $y_2 = y_1 + 1$ from the equation, the inequality gives
$2y_1 + 1 \ge 2$, i.e. $y_1 \ge 0.5$, and the objective $5y_1 + 1$ is smallest at $y_1 = 0.5$: $y = (0.5, 1.5)$,
value 3.5. The two values agree, as strong duality says, and the two positive dual prices match the two binding
primal constraints.

## Exercise 2

With $y \in \{0, 1\}$: $x_1 + x_2 \le 4 + M_1 y$ and $3x_1 + x_2 \le 6 + M_2(1 - y)$; $y = 0$ enforces the first
constraint and relaxes the second, $y = 1$ the other way round. A big-M is valid when the relaxed constraint
excludes no point that the other one allows, so the smallest valid value is the largest violation over the *other*
disjunct, not over the whole box: $M_1 = \max\{x_1 + x_2 - 4 : 3x_1 + x_2 \le 6,\ 0 \le x \le 10\} = 2$ (at
$(0, 6)$) and $M_2 = \max\{3x_1 + x_2 - 6 : x_1 + x_2 \le 4,\ 0 \le x \le 10\} = 6$ (at $(4, 0)$);
`smallest_valid_m()` solves the two small LPs. The box bound alone gives $M_1 = 20 - 4 = 16$ and
$M_2 = 40 - 6 = 34$, also valid but loose. Smaller values are wrong: $M_1 = 1$ forbids $(0, 6)$ and $M_2 = 5$
forbids $(4, 0)$ (`test_either_or.py`). For $\max x_1 + x_2$ the integer optimum is 6 at $(0, 6)$, $y = 1$, for
every valid pair. The LP relaxation ($0 \le y \le 1$) is where $M$ matters (`solve(..., relax=True)`):

| $(M_1, M_2)$ | Relaxation bound | at $x$, $y$ |
|---|---|---|
| (2, 6) | 6.00 | (0, 6), $y = 1$ |
| (16, 34) | 13.37 | (3.37, 10), $y = 0.585$ |
| (160, 340) | 20.00 | (10, 10), $y = 0.100$ |

The plot of the relaxation's projection onto the $x$ plane shows why: with a fractional $y$ both constraints are
shifted outward by $M_1 y$ and $M_2(1 - y)$, so the relaxed region is the union of the two disjuncts *plus* the
area between them, and the larger $M$ the more of the box the shifted lines let in; with $M \to \infty$ the
relaxation is the whole box and the bound is useless. With the smallest valid $M$ the relaxation here is exact
(its hull is the convex hull of the two disjuncts), so branch and bound stops at the root; with the box values it
starts from a bound more than twice the optimum and has to branch on $y$.

## Exercise 3

Bound: the LP relaxation of a knapsack is the greedy fractional solution, items in order of value per unit weight
$c$ (3/1), $a$ (5/2), $b$ (4/3). `branch_and_bound()` prints the tree:

```text
[root]           bound 10.67 = 3 + 5 + 4·(2/3), b fractional -> branch on b
  [b=0]          c = 1, a = 1, weight 3 <= 5: integer, incumbent 8
  [b=1]          capacity left 2: c = 1, a = 1/2 -> bound 9.5 > 8, branch on a
    [b=1, a=0]   c = 1: bound 7 <= 8, pruned by bound
    [b=1, a=1]   weight 5, no room for c: integer, 9 > 8, new incumbent 9
```

The optimum is 9 with $a = b = 1$, $c = 0$; the first incumbent, 8, came from the greedy choice and was beaten by
a solution that leaves out the "best" item $c$. Four nodes were enough for a tree that could have had $2^3$
leaves; the pruning of $[b=1, a=0]$ used the bound, that of $[b=1, a=1, c=1]$ never arose because the node was
already integer. `pulp_check()` with HiGHS returns 9 and the same solution.

## Exercise 4

`action_counts()` draws `[52, 39, 32, 18, 20, 5, 7, 3, 13, 50, 40, 55, 32, 38, 59, 45, 39, 34, 35, 57, 19, 50, 41,
3, 25, 52, 35, 4, 47, 45, 52, 13, 8, 53, 4, 34, 7, 20, 30, 27]` (sum 1 242). `count_workset_size()` mirrors ToOp's
function: the number of split combinations is $\sum_{k=0}^{K} \sum_{|S| = k} \prod_{i \in S} n_i$ (the empty
set, the unsplit grid, included), and it is multiplied by the number of ways to switch off up to 3 of 200 branches,
$\sum_{d=0}^{3} \binom{200}{d} = 1 + 200 + 19\,900 + 1\,313\,400 = 1\,333\,501$. With at most 2 splits the first
factor is $1 + 1\,242 + \sum_{i<j} n_i n_j = 747\,139$ and the total $9.96 \times 10^{11}$ candidates: at $10^5$
evaluations per second, 115 days. With at most 4 splits the product sums reach $8.1 \times 10^{10}$ and the total
$1.08 \times 10^{17}$: 34 000 years. Even at the $10^8$ evaluations per second of a GPU batch the four-split
enumeration takes 34 years; the space grows with the product of the action counts, i.e. exponentially in the
number of simultaneous splits, which is why ToOp samples it (Map-Elites) instead of enumerating it, and why its
brute-force generator is a baseline for small cases only.

## Exercise 5

*In NP.* The decision version asks: is there a subset of items with total weight $\le W$ and total value $\ge V$?
A certificate is the subset itself, $n$ bits. The verifier adds the weights and the values of the chosen items and
compares them with $W$ and $V$: $O(n)$ additions of numbers of the input's size, polynomial in the input length.
A yes-instance has such a certificate and a no-instance has none, so 0-1 knapsack is in NP (`verify_certificate()`
is the verifier).

*The dynamic program.* $T[i, w]$ = best value using the first $i$ items with total weight at most $w$:
$T[0, w] = 0$ and $T[i, w] = \max(T[i-1, w],\ T[i-1, w - w_i] + v_i)$ when $w_i \le w$, else $T[i-1, w]$. The
answer is $T[n, W]$ and the chosen items are read back by walking from $T[n, W]$ and checking at each row whether
the value changed. The table has $n(W+1)$ cells, each filled in constant time: $O(nW)$. `knapsack_dp()` on the
instance of Exercise 3 fills 18 cells and returns 9 with items $\{a, b\}$; for $n = 50$ the running time goes from
0.3 ms at $W = 1\,000$ to 1.2 ms at $W = 10\,000$ and 20 ms at $W = 100\,000$, linear in $W$.

*Why this does not contradict NP-completeness.* The input writes $W$ in binary, with $\log_2 W$ bits (10, 14 and 17
bits in the three runs). A running time of $O(nW) = O(n\,2^{\text{bits of } W})$ is exponential in the length of
the input: doubling the number of bits of $W$ squares the running time. The algorithm is polynomial only in the
*numeric value* of $W$, which is what pseudo-polynomial means; a knapsack with weights of 40 bits is beyond this
table even for ten items. NP-hardness is a statement about input length, so knapsack is NP-complete and
pseudo-polynomially solvable at the same time (it is *weakly* NP-hard; problems like 3-partition that stay hard with
small numbers are strongly NP-hard). The practical reading for topology optimization: an exact method may be fast
when the numbers are small or the structure is special, and hopeless on the general case, so one checks the
structure before trusting a complexity class in either direction.
