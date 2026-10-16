# Chapter 2 – Solutions

Solutions of the Lab and the Exercises of [Chapter 2](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`laplacian.py`](src/laplacian.py) and [`laplacian.m`](src/laplacian.m)
(the worked example: `incidence`, `laplacian`, `nodal_voltages`), [`lab_network.py`](src/lab_network.py) (the
six-node network), [`resistor_lab.py`](src/resistor_lab.py) (Lab steps 1–5 and the checks of Exercises 1, 2, 4
and 5), [`branch_table.py`](src/branch_table.py) (Exercise 6) and [`two_triangles.m`](src/two_triangles.m)
(Exercise 3). `python resistor_lab.py` and `python branch_table.py` print every number quoted here; the Octave
file carries its own `%!test` block and was run with Octave 11.3. Node indices are zero-based as in the lab (one-based
in Octave).

## Lab

1. **Incidence matrix and Laplacian.** With `fr = [0, 0, 1, 1, 2, 3, 4]`, `to = [1, 2, 2, 3, 4, 4, 5]` and the seven
   conductances, $A$ is $7 \times 6$ with one $+1$ and one $-1$ per row, and

    $$
    G = A^\top \operatorname{diag}(g)\, A =
    \begin{pmatrix}
    15 & -10 & -5 & 0 & 0 & 0 \\
    -10 & 22 & -8 & -4 & 0 & 0 \\
    -5 & -8 & 19 & 0 & -6 & 0 \\
    0 & -4 & 0 & 7 & -3 & 0 \\
    0 & 0 & -6 & -3 & 11 & -2 \\
    0 & 0 & 0 & 0 & -2 & 2
    \end{pmatrix} \text{S}.
    $$

    Each diagonal entry is the sum of the conductances at the node ($G_{00} = 10 + 5$), each off-diagonal entry
    minus the conductance of the branch between the two nodes. `resistor_lab.checks` reports `symmetric: True`,
    `max_row_sum: 0.0` and `zero_eigenvalues: 1`; the eigenvalues are $0,\ 1.9877,\ 7.3497,\ 11.1452,\ 23.969,\
    31.5484$. Symmetry follows from $G^\top = A^\top \operatorname{diag}(g)^\top A = G$; the row sums vanish
    because $A\mathbf{1} = 0$ (every row of $A$ has a $+1$ and a $-1$), so $G\mathbf{1} = 0$ and $\mathbf{1}$ is
    an eigenvector with eigenvalue 0. There is only one zero eigenvalue because the network is connected
    (Exercise 2 proves it).

    ```python title="resistor_lab.py"
    --8<-- "docs/part-1-math/ch02-linear-algebra/src/resistor_lab.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch02-linear-algebra/src){ target="_blank" } to see or download all the files of this chapter.

2. **Without branch (4, 5).** Node 5 hangs on that single branch, so deleting it leaves two connected components,
   $\{0, 1, 2, 3, 4\}$ and $\{5\}$. The eigenvalues become $0,\ 0,\ 7.1416,\ 9.5979,\ 23.7563,\ 31.5042$: two
   zero eigenvalues, one per component. `null_space(G)` returns a $6 \times 2$ orthonormal basis: the first vector
   is $0.4472 = 1/\sqrt 5$ on nodes 0–4 and 0 on node 5, the second is $(0, 0, 0, 0, 0, 1)$. Each basis vector is
   constant on each component because $Gx = 0$ means $x^\top G x = \sum_k g_k (x_{f_k} - x_{t_k})^2 = 0$, so
   $x_{f_k} = x_{t_k}$ across every branch, and following branches within a component makes $x$ constant there;
   between components there is no branch, so the two constants are free, and the null space is the span of the two
   indicator vectors. `resistor_lab.components` recovers the components by grouping nodes with equal rows of the
   basis. The figure (`eigenvalue_figure`, two curves) shows the sorted eigenvalues of both cases: the full
   network starts at a single 0 and climbs to 31.5; the cut network starts with two zeros, its third eigenvalue
   (7.14) is below the full network's second non-zero one (7.35) and its largest (31.50) barely moves, since
   the 2 S branch to the leaf node contributes little to the rest of the spectrum.
3. **Nodal solve.** Inject $i = (1, 0, 0, 0, 0, -1)$ A, ground node 0 (drop its row and column), solve
   $G_r v_r = i_r$ and set $v_0 = 0$:

    $$
    v = (0,\ -0.0550,\ -0.0901,\ -0.1222,\ -0.2119,\ -0.7119)\ \text{V}.
    $$

    The branch currents $g \odot (A v)$, in the order of the branch list, are $0.5497,\ 0.4503,\ 0.2807,\ 0.2690,\
    0.7310,\ 0.2690,\ 1.0000$ A. KCL holds at every node, the grounded one included: $\max |A^\top (g \odot A v)
    - i| = 2 \times 10^{-16}$. Reading the currents: the 1 A leaves node 0 split 0.55 / 0.45 between the 10 S and
    5 S branches, the whole ampere must cross branch (4, 5) because it is the only branch into node 5, and node 4
    collects 0.731 A from node 2 and 0.269 A from node 3. The voltage of node 5 is $-0.7119$ V: the effective
    resistance between nodes 0 and 5 is $0.7119\ \Omega$.
4. **Ground node 5 instead.** The solve gives $v = (0.7119,\ 0.6569,\ 0.6218,\ 0.5897,\ 0.5000,\ 0)$ V. The
   difference from step 3 is $0.7119$ V at every node, and the branch currents are identical to $10^{-12}$ A
   (KCL residual $2 \times 10^{-15}$). The reason is the null space of step 1: the solutions of $Gv = i$ form a
   line $v + c\mathbf{1}$, since $G(v + c\mathbf{1}) = Gv$; grounding a node picks one point of the line by
   fixing $c$. Currents depend only on voltage *differences*, $A(v + c\mathbf{1}) = Av$, so they are the same
   for every choice of ground. ToOp's slack bus plays this role for the voltage angles of the DC power flow
   ([Chapter 16](../../part-3-power-systems/ch16-dc-power-flow/index.md)).
5. **A new branch (0, 5).** Its incidence row is $a = (1, 0, 0, 0, 0, -1)$, and `rank_one_update(G, a, 1.0)`
   equals the Laplacian rebuilt from the eight-branch list (`np.allclose` is `True`): $G' = G + g\, a a^\top$ adds
   $g = 1$ to $G_{00}$ and $G_{55}$ and $-1$ to $G_{05}$ and $G_{50}$, nothing else. With the same injection and
   node 0 grounded, $v = (0,\ -0.0321,\ -0.0526,\ -0.0714,\ -0.1238,\ -0.4159)$ V. The new branch carries
   $0.4159$ A, so the old network carries only $0.5841$ A: the old branch currents become $0.3211,\ 0.2630,\
   0.1640,\ 0.1571,\ 0.4270,\ 0.1571,\ 0.5841$ A, each one scaled by $0.5841$ (the change is
   $-0.2286,\ -0.1873,\ -0.1167,\ -0.1119,\ -0.3040,\ -0.1119,\ -0.4159$ A). This is a parallel connection: the
   old network is a $0.7119\ \Omega$ resistor between nodes 0 and 5 (step 3), in parallel with the new
   $1\ \Omega$, giving $0.7119 / 1.7119 = 0.4159\ \Omega$ and the same current split. The rank-1 structure is
   what [Chapter 17](../../part-3-power-systems/ch17-sensitivity-factors-1/index.md) exploits: one outer product
   per topology change, instead of a new factorisation.

## Exercise 1

Number the nodes 0, 1, 2 and the branches $0 \to 1$, $1 \to 2$, $2 \to 0$:

$$
A = \begin{pmatrix} 1 & -1 & 0 \\ 0 & 1 & -1 \\ -1 & 0 & 1 \end{pmatrix},
\qquad
A^\top A = \begin{pmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{pmatrix}.
$$

Each diagonal entry counts the branches at the node (2), each off-diagonal entry is $-1$ because every pair of
nodes shares one branch. The rows sum to zero, so $A^\top A\, \mathbf{1} = 0$; the rank is 2 (subtracting the
first row from the others leaves two independent rows), so the null space is one-dimensional: $\operatorname{span}
\{(1, 1, 1)\}$, normalised $\pm(0.5774, 0.5774, 0.5774)$ (`resistor_lab.triangle()`).

Removing one branch, say $2 \to 0$, leaves a path $0 - 1 - 2$:

$$
A^\top A = \begin{pmatrix} 1 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 1 \end{pmatrix}.
$$

The matrix changes (the end nodes now have degree 1 and no longer touch each other), but the rank is still 2 and
the null space is still $\operatorname{span}\{(1, 1, 1)\}$: the graph is still connected. What changed is the
*rank of $A$ itself*: three rows for three nodes gave rank 2 (the three rows of the triangle sum to zero, a loop),
two rows for a tree give rank 2 with no dependent row. Only removing a *second* branch would disconnect a node and
add a second zero eigenvalue.

## Exercise 2

Write $y = Ax$, so $y_k = x_{f_k} - x_{t_k}$ is the difference across branch $k$. Then

$$
x^\top G x = x^\top A^\top \operatorname{diag}(g)\, A x = (Ax)^\top \operatorname{diag}(g)\,(Ax) = \sum_k g_k y_k^2
= \sum_k g_k (x_{f_k} - x_{t_k})^2 .
$$

(Physically: $x$ a vector of node voltages, the sum is the power dissipated in the resistors.) With every
$g_k > 0$ each term is non-negative, so $x^\top G x \ge 0$ for every $x$: $G$ is positive semidefinite, and all its
eigenvalues are $\ge 0$. If $Gx = 0$ then $x^\top G x = 0$, so every term vanishes: $x_{f_k} = x_{t_k}$ for every
branch $k$. Two nodes joined by a path of branches therefore have the same $x$, which is exactly "constant on each
connected component". Conversely any such vector gives $Ax = 0$ and hence $Gx = 0$, so the null space has
dimension equal to the number of components, and a connected network has exactly one zero eigenvalue
(Lab steps 1 and 2). Numeric check on the lab network with a random $x$: $x^\top G x = 39.2931$ and
$\sum_k g_k (x_{f_k} - x_{t_k})^2 = 39.2931$ (`resistor_lab.quadratic_form`).

## Exercise 3

Two triangles on nodes $1, 2, 3$ and $4, 5, 6$ (Octave indices), unit conductances. `two_triangles()` returns the
$6 \times 6$ Laplacian, block diagonal with two copies of the matrix of Exercise 1, and:

- `eig`: $0,\ 0,\ 3,\ 3,\ 3,\ 3$. **Two zero eigenvalues**, one per component; a unit triangle has eigenvalues
  $0, 3, 3$, and the block structure just repeats them.
- `rank`: **4** $= 6 - 2$, the number of nodes minus the number of components.
- `null`: a $6 \times 2$ basis, each vector constant on each triangle (the test checks that the differences inside
  a triangle vanish); the indicator vectors $(1,1,1,0,0,0)$ and $(0,0,0,1,1,1)$ span the same space.

With other conductances (`two_triangles([1 2 3 4 5 6])`) the non-zero eigenvalues move but the two zeros and the
rank 4 stay: they depend only on connectivity.

```octave title="two_triangles.m"
--8<-- "docs/part-1-math/ch02-linear-algebra/src/two_triangles.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch02-linear-algebra/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`resistor_lab.three_ways` computes `A.T @ np.diag(g) @ A`, `A.T @ (g[:, None] * A)` and
`np.einsum("ki,k,kj->ij", A, g, A)`; on the lab network the three agree (`np.allclose` is `True`). For the timing,
`random_network(2000, 1000)` builds a connected graph of 2 000 nodes and 2 999 branches (a chain plus 1 000 random
branches); `time_three_ways()` measured, on this machine and in a single run (NumPy 2.4.6):

| Method | Time |
|---|---|
| `A.T @ np.diag(g) @ A` | 0.34 s |
| `A.T @ (g[:, None] * A)` | 0.14 s |
| `np.einsum("ki,k,kj->ij", A, g, A)` | 15.2 s |
| `np.einsum(..., optimize=True)` | 0.14 s |

The numbers vary from run to run (another run gave 0.32, 0.14, 16.9 s), the ranking does not. The explicit
`np.diag` builds a $2999 \times 2999$ matrix of 9 million entries, almost all zero, and multiplies it: wasted work
and memory. Broadcasting scales each row of $A$ by $g_k$ in one pass and hands the rest to BLAS. Plain `einsum`
with three operands evaluates the triple sum over $k, i, j$ in one loop of $2999 \times 2000 \times 2000 = 1.2
\times 10^{10}$ multiply-adds without BLAS, a hundred times slower; `optimize=True` lets it contract pairwise,
and it is as fast as the broadcast form. ToOp's `dc_solver/preprocess/helpers/ptdf.py` does the same with SciPy
sparse matrices: `connectivity_matrix.multiply(susceptances[:, np.newaxis])`, the scaled incidence matrix, then its
product with `connectivity_matrix.T`.

## Exercise 5

By linearity, $G' x = (G + g\, a a^\top) x = Gx + g\, a (a^\top x) = Gx + g\,(a^\top x)\, a$, since $a^\top x$ is a
scalar and can be moved in front of $a$. Numerically, `resistor_lab.rank_one_product(G, a, 1.0, x)` for a random
$x$ gives $\max |G'x - (Gx + g (a^\top x) a)| = 0$ to the last bit. The cost: forming $G'$ explicitly is an
$n \times n$ outer product, $O(n^2)$ operations and storage, followed by an $O(n^2)$ matrix–vector product. The
right-hand side needs one inner product $a^\top x$ ($n$ multiplications), one scaling of $a$ ($n$) and one vector
addition ($n$): $O(n)$ on top of $Gx$, which is already available or cheap when $G$ is sparse (an incidence row
$a$ has only two non-zero entries, so even less). This is the arithmetic behind the bus-split update of
`dc_solver/jax/bsdf.py`: the PTDF after a split is the old PTDF plus one outer product, applied to many injection
vectors without ever rebuilding the matrix.

## Exercise 6

`branch_table.branch_frame()` builds the `DataFrame` with columns `from`, `to`, `g` from `lab_network.py`;
`round_trip` writes it with `to_csv(index=False)` and reads it back with `read_csv` (an in-memory `StringIO` in the
script, a file path in the test), and `incidence_from_frame` rebuilds $A$ from the two integer columns: it equals
the original (`np.array_equal` is `True`). `with_currents` adds the branch currents of Lab step 3 and sorts by
absolute value:

| from | to | g (S) | current (A) |
|---|---|---|---|
| 4 | 5 | 2 | 1.000000 |
| 2 | 4 | 6 | 0.730994 |
| 0 | 1 | 10 | 0.549708 |
| 0 | 2 | 5 | 0.450292 |
| 1 | 2 | 8 | 0.280702 |
| 1 | 3 | 4 | 0.269006 |
| 3 | 4 | 3 | 0.269006 |

The last two are equal because branches (1, 3) and (3, 4) are in series through node 3, which has no injection.

```python title="branch_table.py"
--8<-- "docs/part-1-math/ch02-linear-algebra/src/branch_table.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch02-linear-algebra/src){ target="_blank" } to see or download all the files of this chapter.
