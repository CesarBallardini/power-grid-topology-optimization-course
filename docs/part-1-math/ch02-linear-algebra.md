# Chapter 2. Linear algebra for networks

**Weeks:** 4 · **Semester:** 1 · **Prerequisites:** none

!!! info "Why it matters for ToOp"

    The DC solver is linear algebra. Flows are a matrix–vector product $F = \mathrm{PTDF}\, P$. The nodal
    susceptance matrix is a weighted graph Laplacian

    $$
    B = A^\top \operatorname{diag}(b)\, A
    $$

    built from the branch–node incidence matrix $A$ (`dc_solver/preprocess/helpers/ptdf.py`). $B$ is singular
    (its null space is the constant vector), so a slack bus is removed before solving. Every topology change
    is a low-rank correction of the PTDF: a bus split adds one outer product $u v^\top$
    (`dc_solver/jax/bsdf.py`), and a set of $k$ disconnections adds a rank-$k$ term built from a small
    $k \times k$ solve (`dc_solver/jax/disconnections.py`, `dc_solver/jax/multi_outages.py`). This chapter builds the same objects for resistor networks; the power
    system versions come in [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md) and
    [Chapter 17](../part-3-power-systems/ch17-sensitivity-factors-1.md).

## Topics

- Vectors, matrices, matrix–vector and outer products; block matrices; index (einsum) notation.
- Linear systems, Gaussian elimination, LU; inverses; determinants; Cramer's rule for 2×2 and 3×3.
- Vector spaces, rank, null space, the four fundamental subspaces.
- Symmetric and positive (semi)definite matrices; eigenvalues; pseudoinverse; least squares.
- **Incidence matrices of graphs; Kirchhoff's laws in matrix form; the $A^\top C A$ framework.** The weighted
  Laplacian $A^\top \operatorname{diag}(g) A$ first appears here; it is revisited in
  [Chapter 3](ch03-graph-theory.md) as the graph Laplacian and in
  [Chapter 8](../part-2-circuits/ch08-dc-circuits.md) as the nodal conductance matrix.
- Superposition as linearity.

## Learning outcomes

- Build the incidence matrix of a small network and show that $A^\top A$ has a null space of dimension equal to
  the number of connected components.
- Explain why removing one row and column (the slack bus) makes $B$ invertible for a connected grid.
- Compute a rank-1 update of a matrix and its effect on a product.

## Resources

- Courses and videos:
  - 3Blue1Brown, *Essence of Linear Algebra* (start here for intuition):
    <https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab>.
  - MIT OCW 18.06SC (Strang): <https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/>. Unit I
    "Factorization into A = LU" and "Graphs, Networks, Incidence Matrices"; Unit III "Symmetric Matrices
    and Positive Definiteness" and "Left and Right Inverses; Pseudoinverse". Lecture 12 of 18.06 (2010) is
    the graphs lecture.
  - MIT OCW 18.085 (Strang), the $A^\top C A$ framework for springs and circuits:
    <https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/>.
  - Alternatives: edX UTAustinX *LAFF*; Coursera *Matrix Algebra for Engineers* (HKUST, free notes).
- Free texts:
  - Boyd & Vandenberghe, *Introduction to Applied Linear Algebra (VMLS)*, FREE
    <https://web.stanford.edu/~boyd/vmls/>. Main reference of Berkeley EECS 16A; Ch. 7 has incidence
    matrices.
  - Hefferon, *Linear Algebra*, FREE <https://hefferon.net/linearalgebra/>.
  - Margalit & Rabinoff, *Interactive Linear Algebra*, FREE <https://textbooks.math.gatech.edu/ila/>.
  - Axler, *Linear Algebra Done Right* 4th ed., FREE (open access) <https://linear.axler.net/>.
- Textbooks:
  - Lay, *Linear Algebra and Its Applications* (6 syllabi). 2012 Pearson printing **BORROW**
    [`isbn_9781256144380`](https://archive.org/details/isbn_9781256144380).
    ES: *Álgebra lineal y sus aplicaciones* (Pearson; 5th ed. cited at UTN).
  - Strang, *Introduction to Linear Algebra* (6 syllabi, counted with his *Linear Algebra and Its
    Applications*). 4th ed. **PRINT-DISABLED**. "Graphs and Networks" is §8.2 in the 4th ed. and §10.1 in
    the 5th. ES (*Linear Algebra and Its Applications*): *Álgebra lineal y sus aplicaciones* (cited at UTN
    and FIUBA), **USER-UPLOAD (ES)** [`algebra-lineal-y-sus-aplicaciones-gilbert-strang`](https://archive.org/details/algebra-lineal-y-sus-aplicaciones-gilbert-strang).
  - Strang, *Introduction to Applied Mathematics* (1986), Ch. 2 "Equilibrium equations": the $A^\top C A$
    structure of DC power flow. **BORROW**
    [`introductiontoap0000stra`](https://archive.org/details/introductiontoap0000stra).
  - Hoffman & Kunze, *Linear Algebra* (3 syllabi, theory). 2nd ed. 1971 **BORROW**
    [`linearalgebra00hoff_0`](https://archive.org/details/linearalgebra00hoff_0). ES: *Álgebra lineal* (Prentice Hall Hispanoamericana, 1991).
  - Olver & Shakiban, *Applied Linear Algebra* (ETH), chapter on equilibrium and electrical networks. 1st
    ed. **BORROW** [`isbn_2900131473828`](https://archive.org/details/isbn_2900131473828).
- Documentation and tools:
  - NumPy linear algebra (`numpy.linalg`): <https://numpy.org/doc/stable/reference/routines.linalg.html>;
    SciPy `null_space`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.null_space.html>.
  - GNU Octave, "Linear Algebra" and "Basic Matrix Functions" (`eig`, `null`, `rank`, `pinv`):
    <https://docs.octave.org/latest/Linear-Algebra.html>,
    <https://docs.octave.org/latest/Basic-Matrix-Functions.html>.

## Worked example

Take four nodes and four branches given as arrays: branch $k$ goes from node $f_k$ to node $t_k$ with
conductance $g_k$. The incidence matrix $A \in \mathbb{R}^{m \times n}$ has $A_{k f_k} = +1$ and $A_{k t_k} = -1$,
and the weighted Laplacian (nodal conductance matrix) is

$$
G = A^\top \operatorname{diag}(g)\, A,
\qquad
G_{ii} = \sum_{k\ \text{incident to}\ i} g_k,
\qquad
G_{ij} = -\sum_{k:\ \{f_k, t_k\} = \{i, j\}} g_k \quad (i \ne j).
$$

Each row of $A$ sums to zero, so $A\mathbf{1} = 0$ and $G\mathbf{1} = 0$: $G$ is singular, and for a connected
network its null space is exactly the span of $\mathbf{1}$.

```python
import numpy as np

fr = np.array([0, 1, 2, 2])          # from nodes
to = np.array([1, 2, 0, 3])          # to nodes
g = np.array([1.0, 2.0, 1.0, 0.5])   # conductances in S
m, n = len(fr), 4

A = np.zeros((m, n))
A[np.arange(m), fr] = 1.0
A[np.arange(m), to] = -1.0
G = A.T @ np.diag(g) @ A             # same as np.einsum("ki,k,kj->ij", A, g, A)

w, V = np.linalg.eigh(G)
print(np.round(w, 4))                # [0. 0.6106 3.091 5.2984]: one zero eigenvalue
print(V[:, 0])                       # +-[0.5 0.5 0.5 0.5]: the constant vector

i = np.array([1.0, 0.0, 0.0, -1.0])  # inject 1 A at node 0, withdraw at node 3
v = np.zeros(n)
v[1:] = np.linalg.solve(G[1:, 1:], i[1:])   # ground node 0
print(v, g * (A @ v))                # node voltages and branch currents
```

The same in Octave (indices start at 1):

```octave
fr = [1 2 3 3]; to = [2 3 1 4]; g = [1 2 1 0.5]; m = numel(fr); n = 4;
A = zeros(m, n);
A(sub2ind([m n], 1:m, fr)) = 1;
A(sub2ind([m n], 1:m, to)) = -1;
G = A' * diag(g) * A
eig(G)                   % one zero eigenvalue
null(G)                  % constant vector, up to sign
i = [1; 0; 0; -1];
v = [0; G(2:end, 2:end) \ i(2:end)]
```

## Lab

!!! example "Lab: a resistor network from arrays"

    Tools: Jupyter + NumPy (+ SciPy `null_space`) + matplotlib (uses
    [Chapter 30](../part-5-computing/ch30-scientific-python.md) basics), or Octave. No power system package is
    used; the comparison with a power-flow tool is done in
    [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md).

    Use this network of 6 nodes and 7 branches (conductances in siemens):

    ```python
    fr = [0, 0, 1, 1, 2, 3, 4]
    to = [1, 2, 2, 3, 4, 4, 5]
    g  = [10, 5, 8, 4, 6, 3, 2]
    ```

    1. Write `incidence(fr, to, n)` returning $A$, and form $G = A^\top \operatorname{diag}(g) A$. Check that
       $G$ is symmetric, that every row sums to zero and that `np.linalg.eigvalsh(G)` has exactly one zero
       eigenvalue (up to round-off).
    2. Delete branch `(4, 5)` and recompute. Count the zero eigenvalues and compute `null_space(G)`: explain
       why its basis vectors are constant on each connected component. Plot the sorted eigenvalues of both
       cases with matplotlib.
    3. Back on the full network, inject 1 A at node 0 and withdraw 1 A at node 5. Ground node 0 (delete its
       row and column), solve $G_r v_r = i_r$ and compute the branch currents $g \odot (A v)$. Verify KCL:
       $A^\top (g \odot A v) = i$.
    4. Ground node 5 instead. Show that all node voltages shift by the same constant and that the branch
       currents do not change. Relate this to the null space found in step 1.
    5. Add a new branch `(0, 5)` with $g = 1$ S. Show that the new matrix is $G + g\, a a^\top$, where $a$ is the
       new row of $A$, and compare the new currents with the old ones.

## Exercises

1. By hand, write the incidence matrix of a triangle (3 nodes, 3 branches), compute $A^\top A$ and find its
   null space. What changes if one branch is removed? *(Pen and paper)*
2. Show that $x^\top G x = \sum_k g_k (x_{f_k} - x_{t_k})^2$, and deduce that $G$ is positive semidefinite when
   all $g_k > 0$ and that $Gx = 0$ forces $x$ to be constant on each connected component. *(Pen and paper)*
3. Build two disjoint triangles as one network (6 nodes, 6 branches). Compute `eig`, `rank` and `null` of $G$.
   How many zero eigenvalues are there, and what is the rank? *(Octave)*
4. Implement $A^\top \operatorname{diag}(g) A$ three ways: explicit `np.diag`, `A.T @ (g[:, None] * A)` and
   `np.einsum("ki,k,kj->ij", A, g, A)`. Check that they agree and time them for a network with 2 000 nodes.
   *(Jupyter: NumPy)*
5. Let $G' = G + g\, a a^\top$. Show that $G' x = G x + g\,(a^\top x)\, a$ and verify it numerically for a random
   $x$; explain why this costs $O(n)$ instead of $O(n^2)$. *(Jupyter: NumPy)*
6. Store the branch list of the lab in a pandas `DataFrame` with columns `from`, `to`, `g`, write it to CSV and
   rebuild $A$ from the file. Add a column with the branch currents of lab step 3 and sort by absolute current.
   *(Jupyter: pandas + NumPy)*
