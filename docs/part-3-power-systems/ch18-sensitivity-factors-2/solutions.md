# Chapter 18 – Solutions

Solutions of the Lab and the Exercises of [Chapter 18](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`ieee14.py`](src/ieee14.py) (IEEE 14 as a DC grid, with brute-force
PTDFs of split, reduced or re-parametrised grids), [`bsdf.py`](src/bsdf.py) (the BSDF of the worked example),
[`modf.py`](src/modf.py) (MODF, Woodbury, LODF matrix), [`compose.py`](src/compose.py) (ToOp's composition
order), [`compose_toop.py`](src/compose_toop.py) (the comparison with ToOp), [`splits.py`](src/splits.py),
[`woodbury_timing.py`](src/woodbury_timing.py), [`flops.py`](src/flops.py) and [`bridges.py`](src/bridges.py).
Each module's `main()` prints the numbers quoted below; the ToOp numbers were produced with ToOp `ba1874e` on
CPU. The notebooks [examples.ipynb](src/examples.ipynb), [lab.ipynb](src/lab.ipynb) and
[exercises.ipynb](src/exercises.ipynb) import these modules.

IEEE 14 comes from pandapower's `case14` after `rundcpp`; the DC model (per-unit reactances with transformer
taps, loads and generators, bus 1 as slack) reproduces pandapower's DC flows to $2 \cdot 10^{-13}$ MW. Branch
names are `L<from>-<to>` for lines and `T<from>-<to>` for transformers, with 1-based bus numbers; node indices
in the code are 0-based (bus 2 is node 1).

```python title="ieee14.py"
--8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/ieee14.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

## Lab

1. **BSDF against a recomputed PTDF.** Split bus 2 (generator 40 MW, load 21.7 MW; branches L1-2, L2-3, L2-4,
   L2-5): L1-2 and L2-5 stay on bus A with the injections, L2-3 and L2-4 move to bus B. `bsdf_split` of the
   worked example receives the PTDF with a copy of column 2 appended as the new node and returns $H'$ and $F'$.
   Against the PTDF of the physically split grid (`grid.split(1, bus_a).ptdf()`, 15 nodes), the largest
   differences are $8 \cdot 10^{-16}$ in $H'$ and $1 \cdot 10^{-13}$ MW in the flows, and
   $\beta = H'_{:,i} - H'_{:,b}$ holds to $6 \cdot 10^{-16}$. The flows at the station:

    | Branch | Before | After | $\beta$ |
    |---|---|---|---|
    | L1-2 (bus A) | 147.84 MW | 100.12 MW | −0.3812 |
    | L2-3 (bus B) | 70.01 MW | 29.54 MW | −0.3233 |
    | L2-4 (bus B) | 55.15 MW | −29.54 MW | −0.6767 |
    | L2-5 (bus A) | 40.97 MW | 118.42 MW | +0.6188 |

    After the split, bus B has no injection, so L2-3 and L2-4 carry the same 29.54 MW in series, and bus A must
    push everything that arrives through L1-2 plus its own 18.3 MW out through L2-5: $100.12 + 18.3 = 118.42$.

    ```python title="splits.py"
    --8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/splits.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

2. **Cross-coupler flow and reassignment.** KCL at bus A in the unsplit grid: $c = F_{\text{L1-2}} -
   F_{\text{L2-5}} + P_A = 147.84 - 40.97 + 18.3 = 125.17$ MW flow from A to B through the (fictitious)
   coupler, and $F + c\,\beta = H'P$ holds to $3 \cdot 10^{-14}$ MW (`check_split`). Moving the 40 MW
   generator to bus B ($\Delta P_i = -40$, $\Delta P_b = +40$) changes the flows by $-\beta x$: the largest
   difference between $H'\,P_\text{moved} - F'$ and $-40\,\beta$ is $3 \cdot 10^{-14}$ MW, and against a PTDF
   recomputed with the generator on the new node $1 \cdot 10^{-13}$ MW; the largest flow change is 37.79 MW
   (`reassign`).

3. **MODF and Woodbury.** The joint outage of L2-4 and L4-5 has $\det(I - \mathcal H_{OO}) = 0.0802$, and
   $H + \text{MODF}\,H_{O,:}$ matches the PTDF of the grid without the two branches to $3 \cdot 10^{-15}$; the
   Woodbury update with $\alpha = (-1, -1)$ equals the MODF update to $2 \cdot 10^{-16}$. Raising the
   susceptance of transformer T5-6 by 20 % ($\alpha = 0.2$) gives a PTDF that matches the recomputed one to
   $9 \cdot 10^{-16}$; the transformer's flow rises from 42.79 to 45.13 MW. The closed-form $2 \times 2$ MODF of
   Exercise 1 agrees with the general solve to $2 \cdot 10^{-16}$.

    ```python title="modf.py"
    --8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/modf.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

4. **Composition in ToOp's order.** `compose` splits bus 4 (L2-4 and L4-5 on bus A; L3-4, T4-7 and T4-9 on
   bus B), disconnects L6-13 with a one-branch MODF (its PTDF row and its from/to nodes are invalidated, as
   `apply_disconnections` does), builds the LODF matrix on the updated PTDF for the 19 remaining branches and
   adds the outage of busbar 9. `brute_force` recomputes everything from physically modified grids. Largest
   differences: $4 \cdot 10^{-13}$ MW in the N-0 flows, $1 \cdot 10^{-12}$ MW in the N-1 matrix and
   $3 \cdot 10^{-13}$ MW in the busbar-outage flows. The cross-coupler flow of the split is 69.10 MW and the
   worst post-contingency flow is 219.0 MW on L1-2 after the outage of L1-5 (IEEE 14 ships without ratings, so
   this is a flow, not a loading). Two details of the screen:

    - The outage of T7-8 is a bridge (bus 8 hangs on it alone): its LODF column has a zero denominator and its
      N-1 row is NaN. The brute-force reference simply isolates bus 8 and lets the slack absorb its injection,
      so the comparison skips that row.
    - The busbar outage follows `perform_outage_single_busbar`: remove the busbar's injection through its PTDF
      column, then outage its branches with one MODF. In a bus-branch model every branch of the node is in the
      set, so the first attempt always islands the node itself ($\det = 0$) and the retry keeps the first branch
      as a skeleton: for bus 9, L9-14, T4-9 and T7-9 are outaged and L9-10 stays, carrying 0 MW into an empty
      busbar. For bus 8 the single branch T7-8 is the skeleton and nothing is outaged. In ToOp the set is found
      on the node-breaker station, where a busbar is one of several and stub branches are removed beforehand, so
      the first attempt usually succeeds.

    ```python title="compose.py"
    --8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/compose.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

    **With ToOp.** `compose_toop.py` writes ToOp's own IEEE 14 example (`case14_pandapower`), loads it with
    `load_grid` and `run_initial_loadflow`, and compares. ToOp counts loads positive, so its injections and
    flows are the negatives of ours; its PTDF is the same. The unsplit flows of `run_initial_loadflow`
    (`dynamic_information.unsplit_flow`) agree with `ieee14.load_case14().flows()` to $2 \cdot 10^{-13}$ MW
    and the unsplit metrics are `(0.0, 0.0)` (ratings of 9 900 MW). For the bus 4 split followed by the 19
    single outages that the example's masks allow (the stub T7-8 is excluded), `numpy_reference.run_solver`
    (BSDF, then LODF per outage) and our `split` + `n1_screen` agree to $1 \cdot 10^{-12}$ MW, with every
    success flag `True`.

    ```python title="compose_toop.py"
    --8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/compose_toop.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

5. **The other order.** With the disconnection applied first (`split_first=False`) the final PTDF, the N-0
   flows, the N-1 matrix and the busbar-outage flows are identical to those of step 4 ($1 \cdot 10^{-15}$,
   $1 \cdot 10^{-13}$ MW, $7 \cdot 10^{-13}$ MW): both orders produce the PTDF of the same physical grid, and a
   PTDF does not remember how it was built. The intermediate objects differ: $\beta$ changes by up to 0.1138,
   because it is the PEDF from bus A to bus B *in the network the split is applied to*, with or without L6-13;
   and the cross-coupler flow is 69.10 MW when the split comes first (KCL at bus A on the intact grid) but
   71.97 MW when L6-13 is already out. ToOp stores one $\beta$ per split and reuses it in
   `compute_cross_coupler_flows`, so the stored vectors are only valid for the order in which
   `compute_bus_splits` produced them, and `compute_symmetric_batch` keeps that order fixed.

## Exercise 1

Let $O = \{k_1, k_2\}$ and write the $2 \times 2$ block as

$$
\mathcal H_{OO} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \qquad
I - \mathcal H_{OO} = \begin{pmatrix} 1 - a & -b \\ -c & 1 - d \end{pmatrix}, \qquad
\det = (1 - a)(1 - d) - bc,
$$

$$
(I - \mathcal H_{OO})^{-1} = \frac{1}{\det} \begin{pmatrix} 1 - d & b \\ c & 1 - a \end{pmatrix}, \qquad
\text{MODF} = N\,(I - \mathcal H_{OO})^{-1},
$$

with $N = \mathcal H_{:,O}$ except rows $k_1, k_2$, which are $-(I - \mathcal H_{OO})$ so that each outaged
branch loses its own flow (`modf_2x2`). For $b = c = 0$ the two columns reduce to the two single LODF columns
$\mathcal H_{:,k}/(1 - h_{kk})$.

**Cut set.** Orient both branches from the side $S$ to the side $S^c$ that the cut separates. One MW injected at
the from-end of $k_j$ (in $S$) and withdrawn at its to-end (in $S^c$) must cross the cut in full, and the only
branches crossing it are $k_1$ and $k_2$, so the column sums of $\mathcal H_{OO}$ are one:
$\mathbf 1^\top \mathcal H_{OO} = \mathbf 1^\top$, hence $\mathbf 1^\top (I - \mathcal H_{OO}) = 0$ and the
determinant vanishes. A bridge is the case $k = 1$ ($h_{kk} = 1$); for a cut set of two the diagonal entries
are below one, since one MW across $k_1$ partly returns through $k_2$. On IEEE 14, L1-2 and L1-5 are the two
branches of bus 1: $1 - h_{kk} = 0.162$ and $0.611$ (neither is a bridge), the column sums of $\mathcal H_{OO}$
are $[1, 1]$ and $\det = 7 \cdot 10^{-16}$, below ToOp's $10^{-10}$. For comparison, L2-4 with L4-5 (not a cut)
gives $\det = 0.0802$ (`modf.py`, `main()`).

## Exercise 2

IEEE 118 (pandapower `case118`: 118 buses, 186 branches). For $k = 1, 2, 3$ random branches with random
$\alpha \in [-0.5, 0.5]$, the Woodbury update differs from the PTDF recomputed with the changed susceptances by
at most $1.2 \cdot 10^{-14}$. Timing (best of five repeats of 20 runs, this machine, NumPy's bundled BLAS):

| $k$ | Woodbury | recompute |
|---|---|---|
| 1 | 0.07 ms | 17 ms |
| 2 | 0.06 ms | 8.5 ms |
| 3 | 0.06 ms | 7.9 ms |
| 5 | 0.07 ms | 19 ms |
| 10 | 0.08 ms | 20 ms |
| 20 | 0.11 ms | 13 ms |
| 50 | 0.23 ms | 9.5 ms |
| 100 | 1.8 ms | 6.9 ms |
| 150 | 18 ms | 8.1 ms |

Recomputation (7–20 ms, with large run-to-run noise: it builds the incidence matrices and solves a
$117 \times 117$ system with 186 right-hand sides) wins somewhere between $k = 100$ and $k = 150$ of the 186
branches. The flop counts say the same: the update costs about $2 n_l n_b k$, the recomputation about
$\tfrac{2}{3} n_b^3 + 2 n_l n_b^2$, which cross at $k \approx n_b + n_b^2 / (3 n_l) = 118 + 25 \approx 143$.
For the $k \le 3$ of a PST tap change or a few disconnections the update is two to three orders of magnitude
cheaper, which is why ToOp never refactorizes.

```python title="woodbury_timing.py"
--8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/woodbury_timing.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 3

In the split network (PTDF $H'$, injections $P'$ with the station's injection on bus A) the flows are
$F' = H'P'$. Closing a coupler between A and B that carries $c$ MW from A to B is, seen from the split network,
an extra withdrawal of $c$ at A and an extra injection of $c$ at B, so the unsplit flows are

$$
F = H'\left(P' - c\,e_i + c\,e_b\right) = F' - c\left(H'_{:,i} - H'_{:,b}\right).
$$

Rearranged, $F' = F + c\,\beta$ with $\beta = H'_{:,i} - H'_{:,b}$: the post-split PEDF from A to B, i.e. the
flow pattern of one MW injected at A and withdrawn at B in the split network.

That reading gives the sum rule at once. Inject one MW at bus A and take it out at bus B. KCL at bus A: the one
MW leaves A through the bus-A branches, so, counting a branch positive when its from-end is at A (flow out of
A) and negative when its to-end is at A,

$$
\sum_{\ell \in A_\text{from}} \beta_\ell - \sum_{\ell \in A_\text{to}} \beta_\ell = 1 .
$$

On IEEE 14, bus 2 split as in the Lab: $\beta_{\text{L2-5}} - \beta_{\text{L1-2}} = 0.6188 + 0.3812 =
1.000000000000$ (`splits.beta_sum`); `test_splits.py` checks every admissible split of buses 2, 4, 5 and 6. The
same KCL at bus B gives minus one for the bus-B branches oriented out of B, which is why $p_c$ and $\beta$
together conserve power in $H' = H + \beta p_c$.

## Exercise 4

Bus 4 has five branches (L2-4, L3-4, L4-5, T4-7, T4-9), hence $2^4 - 5 - 1 = 10$ admissible two-way splits
with at least two branches per busbar, mirror images counted once (`admissible_splits`). IEEE 14 has no
ratings, so `ieee14.ratings` sets each limit to 1.3 times the base-case flow with a 40 MW floor; the unsplit
grid is then loaded to 76.9 % at worst. `scan_station` gives, sorted by $|c|$:

| Bus A | $\lvert c \rvert$ | Worst post-split loading |
|---|---|---|
| L2-4 + T4-9 | 9.2 MW | 86.3 % |
| L2-4 + L3-4 | 16.8 MW | 92.8 % |
| L2-4 + T4-7 | 21.0 MW | 96.9 % |
| L2-4 + L3-4 + T4-9 | 33.4 MW | 100.3 % |
| L2-4 + T4-7 + T4-9 | 37.6 MW | 115.8 % |
| L2-4 + L4-5 + T4-7 | 40.7 MW | 104.8 % |
| L2-4 + L3-4 + L4-5 | 44.9 MW | 157.7 % |
| L2-4 + L3-4 + T4-7 | 45.2 MW | 106.7 % |
| L2-4 + L4-5 + T4-9 | 52.5 MW | 109.7 % |
| L2-4 + L4-5 | 69.1 MW | 124.9 % |

Plotted with $|c|$ on the x axis and the worst loading on the y axis (`plt.scatter`, one point per row), the
ten points rise from (9 MW, 86 %) to (69 MW, 125 %) with one outlier at (45 MW, 158 %): the split that leaves
the two transformers alone on bus B, which then feed the whole lower-voltage part through T4-7 and T4-9 with
nothing else. The trend has a reason: the split changes every flow by $c\,\beta$, so $|c|$ is the size of the
disturbance the split causes, and small disturbances rarely create overloads. It is not a law: where the
disturbance lands depends on $\beta$, which is what the outlier shows. ToOp's `cross_coupler_flow` penalty
charges the MW by which $|c|$ exceeds the coupler limit of the station, which keeps the optimizer among splits
whose coupler could actually be closed again (the coupler carries $c$ when the station is recoupled) and, as a
by-product, among the gentle splits in the lower left of the plot; the overload metrics, not the penalty,
decide whether the disturbance helped.

## Exercise 5

With $n_l$ branches, $n_b$ nodes, a split station of $d$ branches of which $a$ stay on bus A, $k$ outages,
$n_f$ N-1 cases and $T$ timesteps, counting a multiply-add as two flops and a $k \times k$ LU solve as
$\tfrac{2}{3}k^3$ (`flops.py`):

| Update | Flops | Leading term |
|---|---|---|
| one BSDF | $d\,n_b + 2a\,n_l + 4a + n_l + 2 n_l n_b$ | $2 n_l n_b$ (the rank-1 update $H + \beta p_c$) |
| one $k$-outage MODF | $n_l k + \tfrac{2}{3}k^3 + 2 n_l k^2 + 2 n_l n_b k$ | $2 n_l n_b k$ (the rank-$k$ update) |
| LODF matrix, $n_f$ cases | $2 n_l n_f + n_f$ | $2 n_l n_f$ (one difference and one division per entry) |
| N-1 flow matrix, $T$ timesteps | $2 n_l n_b T + 2 n_l n_f T$ | $2 n_l (n_b + n_f) T$ |
| full PTDF recomputation | $\tfrac{2}{3} n_b^3 + 2 n_l n_b^2$ | |

IEEE 14 ($n_l = 20$, $n_b = 14$, $d = 5$, $a = 2$, $k = 3$, $n_f = 20$, $T = 1$): 738, 2 118, 820 and 1 360
flops against 9 669 for a recomputation. A 1 000-node grid with 1 500 branches, all of them N-1 cases, one day
of hourly timesteps: $3.0 \cdot 10^6$, $9.0 \cdot 10^6$, $4.5 \cdot 10^6$ and $1.8 \cdot 10^8$ flops against
$3.7 \cdot 10^9$ for a recomputation. Every update is a few dense matrix products of fixed shape, and this is
the point: the N-1 flow matrix, the largest item, is $n_f$ independent rank-1 updates of $T$ independent
vectors, the LODF matrix is $n_l n_f$ independent entries, and a MODF is one small solve followed by a product.
Across a batch of topologies nothing is shared: each topology has its own $\beta$, MODF, LODF matrix and
flows, all with the same shapes, so the whole chain is embarrassingly parallel over the batch (a `vmap` in
ToOp). The only sequential dependence is inside one topology: the second split of a station list needs the
PTDF left by the first (`compute_bus_splits`), and the LODF matrix needs the PTDF after all splits and
disconnections. That dependence is short (a handful of splits) and the parallel dimension is long (thousands
of topologies, $n_f$ cases, $T$ timesteps), which is what makes the GPU the right machine for it
([Chapter 32](../../part-5-computing/ch32-jax-in-toop/index.md)).

```python title="flops.py"
--8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/flops.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 6

`bridges.random_grid` builds a random spanning tree on 12 nodes, adds 8 random branches, and hangs a leaf node
on one radial branch $k$ with random susceptances in $[2, 20]$; `one_minus_hkk` computes $1 - h_{kk}$ from a
PTDF obtained by an LU solve. Over 2 000 grids (`survey(2000)`), $1 - h_{kk}$ is exactly zero in 74.2 % of
them; the median of $|1 - h_{kk}|$ is 0 and the maximum $3.3 \cdot 10^{-16}$, a few units in the last place.
In exact arithmetic $h_{kk} = 1$ for a bridge, because the one MW sent across it has no other way; in floating
point the solve returns the row of $H$ with round-off, and whether the subtraction cancels to zero depends on the
bit pattern, not on the physics.

Which of ToOp's tolerances catch the bridge:

| Check | Catches the bridge in |
|---|---|
| LODF, grid connected only if $\lvert 1 - h_{kk} \rvert > 10^{-11}$ | 100 % of the grids |
| BSDF, $\lvert d \rvert \ge 10^{-5}$ | 100 % |
| $2 \times 2$, $3 \times 3$ and LU solves, $\lvert \det \rvert > 10^{-10}$ | 100 % |
| $1 \times 1$ `solve_and_check_det`, `a != 0` exactly | 74.2 % |

Any relative or absolute tolerance far above machine precision ($10^{-11}$, $10^{-10}$, $10^{-5}$) catches
every bridge here, since the round-off is at the $10^{-16}$ level. The exact test of the $1 \times 1$ case
lets one bridge in four through: the division by $a \approx 10^{-16}$ then produces flows of order $10^{16}$
MW instead of a failure flag. This is the "inconsistent islanding tolerances" discrepancy of the chapter, and
the lesson of [Chapter 4](../../part-1-math/ch04-numerical-linear-algebra/index.md): never test a floating-point
result for exact zero; compare it with a tolerance that is large against the round-off of the computation that
produced it and small against the values that mean something.

```python title="bridges.py"
--8<-- "docs/part-3-power-systems/ch18-sensitivity-factors-2/src/bridges.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch18-sensitivity-factors-2/src){ target="_blank" } to see or download all the files of this chapter.
