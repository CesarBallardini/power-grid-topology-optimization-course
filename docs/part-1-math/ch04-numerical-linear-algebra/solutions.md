# Chapter 4 – Solutions

Solutions of the Lab and the Exercises of [Chapter 4](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`sherman_morrison.py`](src/sherman_morrison.py) (the network and the
rank-1 update of the worked example), [`rank_one_lab.py`](src/rank_one_lab.py) (lab steps 1–3),
[`near_bridge.py`](src/near_bridge.py) (lab step 4 and Exercise 6), [`woodbury.py`](src/woodbury.py)
(Exercise 3), [`floating_point.py`](src/floating_point.py) (Exercise 4), [`sparse_fill_in.py`](src/sparse_fill_in.py)
and [`fill_in.m`](src/fill_in.m) (Exercise 5). Each module's `main()` prints every number quoted below; the worked
examples run in [examples.ipynb](src/examples.ipynb). Timings are from one run on the maintainer's laptop and
will differ on yours; the ratios are what matters.

## Lab

1. **Denominators.** With $G_r^{-1}$ computed once, `denominator_table` evaluates $g_k a_k^\top G_r^{-1} a_k$
   for the seven branches (node 0 grounded):

    | Branch | Ends | $g_k$ (S) | $g_k R_{\text{eff},k}$ | $1 - g_k R_{\text{eff},k}$ |
    |---|---|---|---|---|
    | 0 | 0–1 | 10 | 0.754386 | 0.245614 |
    | 1 | 0–2 | 5 | 0.508772 | 0.491228 |
    | 2 | 1–2 | 8 | 0.631579 | 0.368421 |
    | 3 | 1–3 | 4 | 0.701754 | 0.298246 |
    | 4 | 2–4 | 6 | 0.801170 | 0.198830 |
    | 5 | 3–4 | 3 | 0.602339 | 0.397661 |
    | 6 | 4–5 | 2 | 1.000000 | 0.000000 |

    Branch 6, `(4, 5)`, is the only one with a zero denominator (to $10^{-12}$). It is the only branch at node 5,
    so its removal leaves node 5 with no path to ground: the reduced matrix $G_r'$ gets a zero row and column and
    is singular, and the Sherman–Morrison update divides by zero. Every other value lies strictly between 0 and 1,
    as [Exercise 2](#exercise-2) proves it must.
2. **Three ways to remove `(1, 3)`.** With 1 A injected at node 5, the voltages of nodes 1–5 before the removal
   are $(0.054971, 0.090058, 0.122222, 0.211891, 0.711891)$ V and after it $(0.047059, 0.105882, 0.272549,
   0.272549, 0.772549)$ V, whether $G_r'$ is rebuilt and solved, the inverse is updated by Sherman–Morrison, or the
   solution is updated directly with $v' = v + G_r^{-1} a_k\, \dfrac{g_k\, a_k^\top v}{1 - g_k a_k^\top G_r^{-1}
   a_k}$ (`remove_three_ways` and `update_solution`). The three answers agree to $5.6 \times 10^{-17}$, round-off
   of `float64`. Node 3 now hangs off node 4 through branch `(3, 4)` alone and carries no current, so $v_3 = v_4$;
   node 5 rises because the current has lost one of its two paths from node 4 to ground. The direct update is the
   cheapest of the three: the branch's old flow $g_k a_k^\top v$ is re-injected at its ends and spread with the
   old inverse, one matrix-vector product and no new matrix.
3. **Timings on the 40 × 40 grid.** The grid has 1 600 nodes and 3 120 branches with random conductances in
   $[0.5, 2)$ S. For 100 random single-branch removals (`time_removals`), rebuilding the sparse $G_r'$ and solving
   it with `spsolve` took 0.338 s in total, against 0.026 s for one `splu` factorization followed by 100 updates
   of the solution, in which $G_r^{-1} a_k$ is obtained as `lu.solve(a_k)` (two triangular solves) and no inverse
   is ever formed: 12.9 times faster, with a largest disagreement between the two methods of $2.4 \times 10^{-13}$.
   `plot_timings` shows two straight lines: the update line starts at the one-off factorization cost and has a
   much smaller slope, so it wins after the first few removals. This is why ToOp factors once (`spsolve` in
   `compute_ptdf`) and then applies low-rank updates on the GPU for every topology it evaluates.
4. **Near-bridge in `float32` and `float64`.** With the weak branch `(5, 0)` added, `(4, 5)` has a parallel path
   and its denominator becomes positive but small: about $g_w / 2$, because the rest of the network between
   nodes 4 and 5 has conductance $\approx g_w$ and $1 - g_k R_{\text{eff}} = g_{\text{rest}} / (g_k +
   g_{\text{rest}})$. After the removal, node 5 is tied to ground only through $g_w$, so its voltage is exactly
   $1 / g_w$ (`precision_table`):

    | $g_w$ (S) | Denominator | $v_5$ re-solved | $v_5$ `float32` | rel. error `float32` | rel. error `float64` |
    |---|---|---|---|---|---|
    | $10^{-2}$ | $4.965 \times 10^{-3}$ | 100 | 100.00 | $1.1 \times 10^{-5}$ | $1.9 \times 10^{-14}$ |
    | $10^{-4}$ | $5.0 \times 10^{-5}$ | $10^{4}$ | $1.001 \times 10^{4}$ | $9.6 \times 10^{-4}$ | $1.9 \times 10^{-12}$ |
    | $10^{-6}$ | $5.0 \times 10^{-7}$ | $10^{6}$ | $1.049 \times 10^{6}$ | $4.9 \times 10^{-2}$ | $1.6 \times 10^{-10}$ |

    The denominator is a difference of two numbers close to 1, so its absolute error is about the machine epsilon
    of the format ($1.2 \times 10^{-7}$ for `float32`) and its relative error is $\varepsilon / \text{denominator}$:
    $2 \times 10^{-5}$, $2 \times 10^{-3}$ and $0.2$ in `float32` for the three values, which is the order of the
    errors observed in $v_5$. In `float64` ($\varepsilon = 2.2 \times 10^{-16}$) the same ratio stays below
    $10^{-9}$. ToOp evaluates exactly this kind of update for every contingency and every split, with denominators
    that can be as small as the weakest parallel path allows; `float32` would turn near-bridges into answers
    wrong by percent, which is why `jax_enable_x64` is forced on. Its `1e-11` threshold (`jnp.abs(denom) > 1e-11`
    in `dc_solver/jax/lodf.py`) sits about four orders of magnitude above `float64` round-off: a denominator below
    it is indistinguishable from the exact zero of a bridge and is treated as islanding rather than divided by.

## Exercise 1

Let $\beta = v^\top M^{-1} u$, a scalar, and multiply:

$$
(M + u v^\top)\left(M^{-1} - \frac{M^{-1} u v^\top M^{-1}}{1 + \beta}\right)
= I - \frac{u v^\top M^{-1}}{1 + \beta} + u v^\top M^{-1} - \frac{u\,(v^\top M^{-1} u)\, v^\top M^{-1}}{1 + \beta}.
$$

The last term contains the scalar $\beta$, so the three terms in $u v^\top M^{-1}$ collect to

$$
u v^\top M^{-1}\left(1 - \frac{1}{1 + \beta} - \frac{\beta}{1 + \beta}\right) = u v^\top M^{-1} \cdot 0,
$$

and the product is $I$. The right-hand side is a right inverse of $M + u v^\top$, hence the inverse.

When $1 + \beta = 0$ the formula divides by zero, and rightly so: $(M + u v^\top)\, M^{-1} u = u + u\,\beta =
(1 + \beta)\, u = 0$ with $M^{-1} u \ne 0$ (as $u \ne 0$), so $M + u v^\top$ has a non-trivial null vector and is
singular. There is no inverse to compute. In the resistor network, $u = -g_k a_k$ and $v = a_k$, so
$1 + \beta = 1 - g_k a_k^\top G_r^{-1} a_k$, and the singular case is the bridge of lab step 1 (`bridge_indicator`
returns 0 for branch 6, and `remove_branch` raises `ZeroDivisionError`). `sherman_morrison.py`'s test
`test_rank_one_update_matches_fresh_inverse` is the numerical version of this identity for the six non-bridge
branches.

## Exercise 2

Write $G_r = G_{\text{rest}} + g_k a_k a_k^\top$, where $G_{\text{rest}}$ is the reduced Laplacian of the network
without branch $k$.

*Circuit argument.* $a_k^\top G_r^{-1} a_k$ is the effective resistance $R_{\text{eff},k}$ between the two ends
of branch $k$ (if one end is the grounded node, between the other end and ground). Between those two terminals the
network is branch $k$, of resistance $r_k = 1/g_k$, in parallel with the rest of the network, of effective
resistance $R_{\text{rest}} \in (0, \infty]$: finite when the rest still joins the two ends, infinite when it does
not. The parallel rule gives

$$
R_{\text{eff},k} = \frac{r_k R_{\text{rest}}}{r_k + R_{\text{rest}}} \le r_k,
\qquad\text{so}\qquad
g_k R_{\text{eff},k} = \frac{R_{\text{rest}}}{r_k + R_{\text{rest}}} \le 1,
$$

with equality exactly when $R_{\text{rest}} = \infty$, that is, when no other path joins the ends of branch $k$,
which is the definition of a bridge (the ground counts as a node: a bridge to the grounded node islands the far
side from ground).

*Algebraic argument.* If $G_{\text{rest}}$ is invertible, Sherman–Morrison applied to $G_r = G_{\text{rest}} +
g_k a_k a_k^\top$ gives, with $x = g_k\, a_k^\top G_{\text{rest}}^{-1} a_k \ge 0$ (positive definiteness),

$$
g_k\, a_k^\top G_r^{-1} a_k = x - \frac{x^2}{1 + x} = \frac{x}{1 + x} < 1 .
$$

If $G_{\text{rest}}$ is singular, removing branch $k$ has islanded a set $S$ of ungrounded nodes; let $z$ be its
indicator vector, so $G_{\text{rest}} z = 0$ and $a_k^\top z = \pm 1$ (exactly one end of $k$ is in $S$). Then
$G_r z = g_k a_k (a_k^\top z) = \pm g_k a_k$, so $G_r^{-1} a_k = \pm z / g_k$ and $g_k a_k^\top G_r^{-1} a_k =
(a_k^\top z)^2 = 1$. Thus $g_k R_{\text{eff},k} = 1$ if and only if branch $k$ is a bridge, and the denominator
$1 - g_k R_{\text{eff},k}$ of the worked example vanishes exactly on bridges. Lab step 1 shows the strict
inequality on the six other branches, with values from 0.51 to 0.80.

## Exercise 3

Removing branches $j$ and $k$ changes the reduced Laplacian by $-U C U^\top$ with $U = [a_j\ a_k]$ and
$C = \operatorname{diag}(g_j, g_k)$. Woodbury with $M = G_r$ and $V = U$ gives

$$
(G_r - U C U^\top)^{-1} = G_r^{-1} + G_r^{-1} U \left(C^{-1} - U^\top G_r^{-1} U\right)^{-1} U^\top G_r^{-1},
$$

so the only new system is the $2 \times 2$ one with matrix $C^{-1} - U^\top G_r^{-1} U$ and right-hand side
$U^\top G_r^{-1}$ (two rows). `solve2x2` in `woodbury.py` solves it in closed form, adjugate over determinant, and
returns the solution with a Boolean flag `|det| > 1e-10`; `remove_two_branches` assembles the update. For branches
`(1, 2)` and `(3, 4)` the result matches the inverse rebuilt without them to $1.7 \times 10^{-16}$, and for all
11 pairs of non-bridge branches that leave the network connected the test passes at $10^{-12}$. Two successive
rank-1 updates give the same matrix as one rank-2 update.

Four pairs are singular: `(0, 1)` with `(0, 2)` cuts the whole network from ground, `(1, 3)` with `(2, 4)` cuts
nodes 3–5 off, `(1, 3)` with `(3, 4)` isolates node 3 and `(2, 4)` with `(3, 4)` isolates nodes 4 and 5. For the
last pair the $2 \times 2$ matrix is

$$
\begin{pmatrix} 0.033138 & -0.066277 \\ -0.066277 & 0.132554 \end{pmatrix},
\qquad \det = 9.2 \times 10^{-19},
$$

whose second row is $-2$ times the first: the two branches are the only connections of the island, so their
flows after removal are not independent. The flag is `False` and the "solution" (infinities) is to be ignored.
Any pair that includes the bridge `(4, 5)` is singular too.

ToOp's `solve2x2` (`dc_solver/jax/unrolled_linalg.py`) is the same code in JAX: `det = m1 * m4 - m2 * m3`,
`a_inv = jnp.array([[m4, -m2], [-m3, m1]]) * (1.0 / det)`, `res = jnp.dot(a_inv, b)`, and it returns `(res,
jnp.abs(det) > 1e-10)`. It never raises, because it runs vectorized over thousands of topologies on the GPU, where
an exception in one lane is not an option: the Boolean flag travels with the result and the caller masks out the
islanding cases. `solve3x3` does the same for the $3 \times 3$ case, which is why the chapter says ToOp uses
"closed-form 2×2/3×3 inverses plus determinant checks".

## Exercise 4

`np.finfo` (`finfo_table`):

| | `float32` | `float64` |
|---|---|---|
| mantissa bits | 23 | 52 |
| machine epsilon | $1.19 \times 10^{-7}$ | $2.22 \times 10^{-16}$ |
| decimal digits | 6 | 15 |
| smallest normal | $1.18 \times 10^{-38}$ | $2.23 \times 10^{-308}$ |
| largest | $3.40 \times 10^{38}$ | $1.80 \times 10^{308}$ |

The three expressions (`surprises`):

- `0.1 + 0.2 == 0.3` is `False`: none of the three decimals is representable in binary, and the rounded sum is
  `0.30000000000000004`, one unit in the last place above the rounded `0.3`.
- `1e16 + 1 - 1e16` is `0.0`: near $10^{16}$ the spacing of `float64` numbers is 2, so adding 1 rounds back to
  $10^{16}$ before the subtraction. One decade lower, `1e15 + 1 - 1e15` is `1.0`.
- `np.sum([1.0, np.nan])` is `nan`: NaN is absorbing. `nan_spreads` shows a single NaN entry turning a sum, a
  mean and a max into NaN and poisoning every entry of a dense matrix-vector product.

`compute_ptdf` (`dc_solver/preprocess/helpers/ptdf.py`) ends with `assert np.isnan(ptdf).sum() == 0` because the
PTDF is obtained from a sparse solve with the node susceptance matrix: if the grid is not connected the matrix is
singular and the solver returns infinities or NaNs instead of raising; if some susceptance is NaN, very small or
very large, the same happens. Once a NaN is in the PTDF, every later product (LODF, flows, metrics for every
topology) is NaN too, silently, and an optimizer comparing NaN objectives makes arbitrary choices. Checking once
at the source, where the message can still say what went wrong, is far cheaper than finding the poison downstream.
The assertion's message names exactly these causes.

## Exercise 5

`fill_in_table` builds $G_r$ of the 40 × 40 grid as CSC (1 599 × 1 599, 7 835 nonzeros: the diagonal plus two
entries per branch not at the grounded node) and factors it with `splu`:

| Ordering | nnz(L) + nnz(U) | Fill ratio |
|---|---|---|
| $G_r$ itself | 7 835 | 1.00 |
| `NATURAL` | 126 319 | 16.1 |
| `COLAMD` | 63 291 | 8.1 |
| `MMD_AT_PLUS_A` | 38 325 | 4.9 |

(The unit diagonal of $L$ is not counted.) In the natural ordering the grid's bandwidth of 40 fills the whole band
during elimination: the factors are 16 times denser than the matrix. COLAMD halves that. The third row, not asked
for, is SuperLU's ordering for matrices with a symmetric pattern and does better still; COLAMD is designed for
unsymmetric matrices and does not exploit the symmetry of $G_r$.

Octave (`fill_in.m`, same grid, Cholesky instead of LU): `nnz(G)` is 7 835, `nnz(chol(G))` is 63 959 (8.2 times
the matrix) and, after `p = symamd(G)`, `nnz(chol(G(p, p)))` is 19 955 (2.5 times). A Cholesky factor is one
triangle, so compare it with half of $L + U$; `symamd` lands near SuperLU's symmetric ordering. `spy(G)`,
`spy(chol(G))` and `spy(chol(G(p, p)))` show the banded matrix, the filled band and the sparse scattered factor.
For the 10 × 10 grid the three counts are 455, 989 and 601. The lesson for ToOp: `spsolve` on the node
susceptance matrix pays this fill-in once; the GPU updates that follow touch only dense small blocks and never
refactor.

## Exercise 6

The matrix before the removal is the flat curve: `condition_numbers(weak)` gives $\kappa_2(G_r)$ from 25.83 at
$g_w = 10^{-2}$ to 26.00 at $10^{-6}$, the value of the network without the weak branch, with a log–log slope
of $0.000$. While branch `(4, 5)` is present, node 5 is well connected and the matrix does not care about the
tiny extra conductance. The ill-conditioned one is $G_r'$, the matrix after removing `(4, 5)`, whose smallest
eigenvalue is about $g_w$ (node 5 hanging on the weak branch). `condition_numbers(weak, removed=True)` gives

| $g_w$ (S) | $\kappa_2(G_r)$ | $\kappa_2(G_r')$ | `float32` rel. error of $v_5$ |
|---|---|---|---|
| $10^{-2}$ | 25.83 | $2.99 \times 10^{3}$ | $1.1 \times 10^{-5}$ |
| $10^{-3}$ | 25.98 | $2.99 \times 10^{4}$ | $3.2 \times 10^{-4}$ |
| $10^{-4}$ | 25.99 | $2.99 \times 10^{5}$ | $9.6 \times 10^{-4}$ |
| $10^{-5}$ | 26.00 | $2.99 \times 10^{6}$ | $1.4 \times 10^{-3}$ |
| $10^{-6}$ | 26.00 | $2.99 \times 10^{7}$ | $4.9 \times 10^{-2}$ |

The log–log slope of $\kappa_2(G_r')$ is $-1.000$: the condition number is inversely proportional to the weak
conductance. The `float32` error has a slope of about $-0.9$ over nine points between $10^{-6}$ and $10^{-2}$
(it is noisy, since round-off depends on the particular bits of each value), consistent with the rule of thumb
"relative error $\approx \kappa\, \varepsilon$": at $g_w = 10^{-6}$, $\kappa_2(G_r') \varepsilon_{32} = 3 \times
10^{7} \times 1.2 \times 10^{-7} \approx 3.6$, so `float32` has no correct digits left in the worst direction,
and the 5 % error in $v_5$ is a lucky component. With `float64`, $\kappa \varepsilon_{64} \approx 7 \times 10^{-9}$
and the observed $1.6 \times 10^{-10}$ is comfortably inside. The Sherman–Morrison denominator is the bridge
between the two views: $1 - g_k R_{\text{eff}} \approx g_w / 2$ shrinks in step with the smallest eigenvalue of
$G_r'$, so a small denominator and a large $\kappa_2(G_r')$ are the same warning. `plot_condition` draws
the three curves of the table on log–log axes.
