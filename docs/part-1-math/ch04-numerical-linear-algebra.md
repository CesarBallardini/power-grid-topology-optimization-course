# Chapter 4. Numerical linear algebra and floating point

**Weeks:** 3 · **Semester:** 1 · **Prerequisites:** [Chapter 2](ch02-linear-algebra.md)

!!! info "Why it matters for ToOp"

    ToOp builds its sensitivity matrix (the PTDF, defined in
    [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md)) once with a sparse direct solve
    (`spsolve` in `dc_solver/preprocess/helpers/ptdf.py`). After that, every topology change is a **low-rank
    update** (Sherman–Morrison–Woodbury) evaluated on the GPU, never a refactorization; the concrete updates
    (LODF, BSDF) are derived in Chapters [17](../part-3-power-systems/ch17-sensitivity-factors-1.md) and
    [18](../part-3-power-systems/ch18-sensitivity-factors-2.md). Small k×k systems are solved with LU or
    closed-form 2×2/3×3 inverses plus determinant checks (`dc_solver/jax/unrolled_linalg.py`). Near-singular
    denominators (`jnp.abs(denom) > 1e-11` in `dc_solver/jax/lodf.py`) signal islanding. ToOp forces `float64` in
    JAX (`jax_enable_x64` in `optimizer/dc/main.py`) and pads integer arrays with the width-dependent `int_max()`
    sentinel (`dc_solver/jax/types.py`), so integer widths matter too. This chapter studies the same mechanics on
    a resistor network.

## Topics

- LU with pivoting, Cholesky; determinants from factorizations.
- Norms, condition numbers, stability; ill-conditioning vs algorithmic instability.
- **Sparse matrices**: COO/CSR/CSC storage, fill-in, ordering, sparse direct solvers (SuperLU, KLU,
  UMFPACK).
- **Sherman–Morrison and Woodbury identities; Schur complement (Kron reduction).**
- Direct vs iterative solvers (CG, GMRES) at awareness level.
- Floating-point representation; `float32` vs `float64`; tolerances; NaN propagation.

## Learning outcomes

- Derive the Sherman–Morrison formula and use it to update an inverse after a rank-1 change.
- Explain why a solver that factors once and applies low-rank updates beats refactorization for many small
  changes, and what a zero denominator means for the network.
- Measure how `float32` round-off grows as a denominator approaches zero.
- Choose storage formats and solvers for a sparse symmetric system.

## Resources

- Courses: MIT OCW 18.335J (S. G. Johnson)
  <https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/>; Cornell CS 4220
  (Bindel) <https://www.cs.cornell.edu/courses/cs4220/2023sp/>; fast.ai *Computational Linear Algebra*
  notebooks <https://github.com/fastai/numerical-linear-algebra>.
- Free texts:
  - Driscoll & Braun, *Fundamentals of Numerical Computation*: §2.4 LU, §2.6 pivoting, §2.7 norms, §2.8
    conditioning, §2.9 Cholesky and structure, §7.4 symmetry and definiteness, §8.1 sparsity. FREE
    <https://fncbook.com/>.
  - Higham, "What Is" articles: [Sherman–Morrison–Woodbury](https://nhigham.com/2020/09/29/what-is-the-sherman-morrison-woodbury-formula/),
    [condition number](https://nhigham.com/2020/03/19/what-is-a-condition-number/),
    [LU](https://nhigham.com/2021/04/20/what-is-an-lu-factorization/),
    [Cholesky](https://nhigham.com/2020/08/11/what-is-a-cholesky-factorization/),
    [sparse matrix](https://nhigham.com/2020/09/08/what-is-a-sparse-matrix/),
    [Schur complement](https://nhigham.com/2023/06/01/what-is-the-schur-complement-of-a-matrix/).
  - Saad, *Iterative Methods for Sparse Linear Systems* 2nd ed., Ch. 3 (sparse storage and graphs): FREE
    <https://www-users.cse.umn.edu/~saad/IterMethBook_2ndEd.pdf>. Cited at FIUBA.
  - Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991): FREE
    <https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html>.
  - SciPy sparse tutorial: <https://docs.scipy.org/doc/scipy/tutorial/sparse.html>.
- Textbooks:
  - Golub & Van Loan, *Matrix Computations*: §2.1 (Sherman–Morrison–Woodbury), Ch. 2 sensitivity, Ch. 3 LU,
    §4.2 Cholesky. 3rd ed. 1996 **BORROW** [`bwb_Y0-AAK-944`](https://archive.org/details/bwb_Y0-AAK-944).
  - Trefethen & Bau, *Numerical Linear Algebra* (required at MIT 18.335J): Lectures 12 and 20–23.
    **NOT ON IA**.
  - Press et al., *Numerical Recipes in C* 2nd ed.: LU, sparse systems including Sherman–Morrison and
    Woodbury, Cholesky. **BORROW**
    [`numericalrecipes0000unse_v2g0`](https://archive.org/details/numericalrecipes0000unse_v2g0).
- Documentation and tools: NumPy [`finfo`](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html);
  SciPy [`splu`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.splu.html);
  GNU Octave "Sparse Matrices" <https://docs.octave.org/latest/Sparse-Matrices.html>.

## Worked example

For an invertible $M$ and vectors $u, v$ with $1 + v^\top M^{-1} u \ne 0$, the Sherman–Morrison identity is

$$
(M + u v^\top)^{-1} = M^{-1} - \frac{M^{-1} u\, v^\top M^{-1}}{1 + v^\top M^{-1} u},
$$

and its rank-$k$ generalization (Woodbury) is

$$
(M + U C V^\top)^{-1} = M^{-1} - M^{-1} U \left(C^{-1} + V^\top M^{-1} U\right)^{-1} V^\top M^{-1}.
$$

Take the grounded conductance matrix $G_r$ of [Chapter 2](ch02-linear-algebra.md) (node 0 removed) and let
$a_k$ be row $k$ of the incidence matrix without column 0. Removing branch $k$ gives $G_r' = G_r - g_k a_k
a_k^\top$, so with $u = -g_k a_k$ and $v = a_k$:

$$
(G_r')^{-1} = G_r^{-1} + \frac{g_k\, G_r^{-1} a_k a_k^\top G_r^{-1}}{1 - g_k\, a_k^\top G_r^{-1} a_k}.
$$

The quantity $a_k^\top G_r^{-1} a_k$ is the effective resistance between the two ends of branch $k$
([Chapter 3](ch03-graph-theory.md)). The denominator $1 - g_k R_{\text{eff},k}$ is zero exactly when the branch is
the only path between its ends, that is, when it is a bridge and its removal islands part of the network.

```python
import numpy as np

fr = np.array([0, 0, 1, 1, 2, 3, 4]); to = np.array([1, 2, 2, 3, 4, 4, 5])
g = np.array([10, 5, 8, 4, 6, 3, 2.0]); m, n = len(fr), 6
A = np.zeros((m, n)); A[np.arange(m), fr] = 1; A[np.arange(m), to] = -1
Ar = A[:, 1:]                                  # ground node 0
Gr = Ar.T @ np.diag(g) @ Ar
Minv = np.linalg.inv(Gr)

k = 3                                          # branch (1, 3)
a = Ar[k]
u = Minv @ a
den = 1 - g[k] * (a @ u)
Minv_sm = Minv + g[k] * np.outer(u, u) / den   # Gr is symmetric, so Minv a a^T Minv = u u^T

keep = np.arange(m) != k
Gk = Ar[keep].T @ np.diag(g[keep]) @ Ar[keep]
print(np.max(np.abs(Minv_sm - np.linalg.inv(Gk))))   # ~1e-16
print(1 - g[6] * (Ar[6] @ Minv @ Ar[6]))             # bridge (4, 5): 0
```

```octave
% Ar, g and Minv = inv(Gr) built as above; k = 4 selects the fourth row, branch (1, 3)
a = Ar(k, :)';             % row k as a column vector
u = Minv * a;
den = 1 - g(k) * (a' * u);
Minv_sm = Minv + g(k) * (u * u') / den;
```

## Lab

!!! example "Lab: rank-1 updates of a resistor network"

    Tools: Jupyter + NumPy + SciPy + pandas + matplotlib (uses
    [Chapter 30](../part-5-computing/ch30-scientific-python.md) basics), or Octave. Start from the network of the
    [Chapter 2](ch02-linear-algebra.md) lab (6 nodes, 7 branches, node 0 grounded, 1 A injected at node 5 and
    withdrawn at node 0).

    1. For every branch $k$, compute the denominator $1 - g_k a_k^\top G_r^{-1} a_k$ and collect branch, ends,
       $g_k$, $g_k R_{\text{eff},k}$ and the denominator in a pandas `DataFrame`. Which branch has a zero
       denominator, and what does its removal do to the network?
    2. Remove branch `(1, 3)` three ways: (a) rebuild $G_r'$ and solve again; (b) update the inverse with
       Sherman–Morrison; (c) update the solution directly, $v' = v + G_r^{-1} a_k \dfrac{g_k\, a_k^\top v}{1 - g_k
       a_k^\top G_r^{-1} a_k}$, without forming any new inverse. Check that all three agree to round-off.
    3. Build a larger network, for example the edges of
       `nx.convert_node_labels_to_integers(nx.grid_2d_graph(40, 40))` with random conductances.
       Time "rebuild and solve" against "one factorization plus Sherman–Morrison" for 100 single-branch
       removals and plot the timings.
    4. Near-bridge: add a weak branch `(5, 0)` with $g = 10^{-2}, 10^{-4}, 10^{-6}$ S, so that `(4, 5)` is no longer
       a bridge. For each value, compute the denominator for removing `(4, 5)` and the voltage at node 5 after
       the removal, with the Sherman–Morrison update computed entirely in `float32` (matrix, inverse and
       vectors) and entirely in `float64`, and compare with the
       re-solved `float64` answer. You should see `float32` relative errors grow from about $10^{-5}$ to a few
       percent while `float64` stays below $10^{-9}$. Relate this to ToOp's `float64` policy and its `1e-11`
       denominator threshold.

## Exercises

1. Verify the Sherman–Morrison identity by multiplying $(M + uv^\top)$ by the right-hand side. What happens when
   $1 + v^\top M^{-1} u = 0$? *(Pen and paper)*
2. Show that for a connected network with a grounded node, $g_k\, a_k^\top G_r^{-1} a_k \le 1$, with equality if
   and only if branch $k$ is a bridge. *Hint:* series and parallel resistances. *(Pen and paper)*
3. Remove two branches at once with the Woodbury identity ($U$ has two columns, so you solve a 2×2 system).
   Implement the 2×2 solve in closed form with a determinant check, then compare with `solve2x2` in ToOp's
   `dc_solver/jax/unrolled_linalg.py`. *(Jupyter: NumPy)*
4. Floating point: print `np.finfo(np.float32)` and `np.finfo(np.float64)`; evaluate `0.1 + 0.2 == 0.3`,
   `1e16 + 1 - 1e16` and `np.sum([1.0, np.nan])`. Explain why ToOp's `compute_ptdf` ends with an assertion
   that the result has no NaNs (`dc_solver/preprocess/helpers/ptdf.py`). *(Jupyter: NumPy)*
5. Sparse storage and fill-in: for the 40×40 grid network, store $G_r$ as CSC, factor it with
   `scipy.sparse.linalg.splu` using `permc_spec="NATURAL"` and `"COLAMD"`, and compare `nnz` of the factors
   with `nnz` of $G_r$. In Octave, use `spy`, `nnz` and `symamd` for the same comparison.
   *(Jupyter: SciPy, or Octave)*
6. Plot `np.linalg.cond(G_r)` against the weak conductance of lab step 4 on log–log axes and relate the slope
   to the observed `float32` error. *(Jupyter: NumPy + matplotlib)*
