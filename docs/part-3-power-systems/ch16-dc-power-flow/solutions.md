# Chapter 16 – Solutions

Solutions of the Lab and the Exercises of [Chapter 16](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`dc_power_flow.py`](src/dc_power_flow.py) (the worked example),
[`dc_vs_ac.py`](src/dc_vs_ac.py) (Lab steps 2 and 3), [`toop57.py`](src/toop57.py) (Lab steps 1 and 4, runs in
ToOp's environment), [`matpower_dcpf.m`](src/matpower_dcpf.m) (Lab step 5), [`dc_approximation.py`](src/dc_approximation.py)
(Exercises 1 and 2), [`dc_sparse_solve.py`](src/dc_sparse_solve.py) (Exercise 3), [`makebdc_shift.m`](src/makebdc_shift.m)
(Exercise 4) and [`psdf.py`](src/psdf.py) (Exercise 5). Every `main()` prints the numbers quoted below; the ToOp
numbers were produced with ToOp `ba1874e` on CPU, the timings on one laptop. The Lab steps are also in
[lab.ipynb](src/lab.ipynb) and the exercise statements in [exercises.ipynb](src/exercises.ipynb).

## Lab

1. **ToOp's example notebook.** `toop57.py` does what `example1_dc_loadflow_example.ipynb` does:
   `case57_data_powsybl(folder)` writes the grid and masks, `load_grid(..., pandapower=False)` preprocesses them and
   `run_initial_loadflow` fills `unsplit_flow`. Its shape is `(1, 82)` = `(n_timesteps, n_branches)`: one injection
   snapshot and 82 branches (64 lines and 18 two-winding transformers, because `case57_data_powsybl` extends
   pypowsybl's IEEE 57 with a phase-shifting transformer and its auxiliary line and bus), on 72 nodes (the
   example also adds busbars, `create_busbar_b_in_ieee`). `nodal_injections` has the matching shape `(1, 72)`. The
   first axis is the time axis of the note in the chapter: a second snapshot would add a row to both arrays and
   leave the PTDF alone. `branch_names` are node-pair labels (`'B1 ## B2 ## '`) and repeat for parallel branches
   (80 distinct names for 82 branches); `branch_ids` are the pypowsybl ids and are unique.

    ```python title="toop57.py"
    --8<-- "docs/part-3-power-systems/ch16-dc-power-flow/src/toop57.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch16-dc-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

2. **DC flows against `rundcpp` on `case57`.** `dc_vs_ac.dc_case` reads pandapower's internal MATPOWER-style
   branch table (`net._ppc['branch']`: from, to, `BR_X`, `TAP`, `SHIFT` and the flow `PF` in MW) after `rundcpp`,
   and the injections as `-net.res_bus.p_mw`, which are balanced because the DC model has no losses. The case
   has 80 branches, 15 of them transformers with a tap ratio, no phase shift, slack at bus 0. With $b = 1/x$ the
   worked example's function differs from `rundcpp` by up to 1.286 MW, all of it on transformers (the largest on
   `trafo 12`, tap 0.895: $-30.29$ against $-31.57$ MW). With $b = 1/(x\tau)$ the difference is
   $2.2 \times 10^{-13}$ MW. That is the systematic difference to find: MATPOWER's and pandapower's DC model
   divide the series susceptance by the tap ratio (manual eq. 3.23), and a tap of 0.895 changes the branch
   susceptance by 12 %. The per-unit base is pandapower's `sn_mva` (100 MVA), already applied inside `_ppc`;
   the orientation is pandapower's own, from bus to to bus for lines and hv to lv for transformers, which is why
   `res_line.p_from_mw` and `res_trafo.p_hv_mw` are the columns to compare. Had the case contained a shift,
   `SHIFT` would have appeared as the extra term $P_{f,\text{shift}}$ of the worked example (Exercise 4).

    ```python title="dc_vs_ac.py"
    --8<-- "docs/part-3-power-systems/ch16-dc-power-flow/src/dc_vs_ac.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch16-dc-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

3. **DC against AC.** `dc_ac_frame('case57')` adds `runpp`'s flows and losses. The AC solution loses 30.29 MW of the
   695.8 MW of load; the DC flows miss the AC ones by 1.56 MW on average and by 10.18 MW at most, on `line 14`
   (139.4 MW DC against 149.6 MW AC). Among the branches above 10 MW the median relative error is 5.3 % and the
   largest 27.5 % (`line 21`, a lightly loaded line whose AC flow is far from its DC value). The absolute error
   correlates with the magnitude of the AC flow (coefficient 0.62): the heavier the branch, the more losses and
   voltage drop it carries, and both are absent from the DC model. The relative error does **not** correlate
   with $r/x$ on this case (coefficient $-0.02$): the lines of `case57` have $r/x$ between 0 and 1.09, and the
   transformers have $r = 0$; the error is set by the voltage profile (0.94 to 1.06 pu, Exercise 1) and by how
   the 30 MW of losses are routed to the slack, not by the resistance of each branch. On a case with a wide
   spread of $r/x$ (distribution grids) the second plot would show the trend the exercise expects. The five
   worst absolute errors are all lines carrying 30 to 150 MW. A scatter of `rel_error` against `|p_ac|` is a
   funnel: large relative errors only on small flows; against `r_over_x` it is a cloud.
4. **ToOp's `unsplit_flow`.** On ToOp's extended IEEE 57, `unsplit_flow` equals
   `network_data.basecase_dc_branch_flows` to $1.6 \times 10^{-13}$ MW (the default `ac_dc_interpolation = 0`
   adds no mismatch) and equals **minus** pypowsybl's `p1` from `run_dc` with the same `CGMES_DISTRIBUTED_SLACK`
   parameters on the saved grid, to $10^{-4}$ MW: `get_basecase_dc_branch_flows` negates `p1` because
   PowSyBl's convention is opposite to the solver's from-node to to-node orientation. The largest flows are
   $-271.2$ MW on `L8-9-1`, 113.9 MW on `L7-8-1`, $-73.2$ MW on `T7-29-1`, 70.9 MW on `L6-8-1` and $-50.1$ MW on
   `L1-17-1`. The backend also stores the AC–DC mismatch $m = F^{AC} - F^{DC}$ per branch: 1.51 MW on average,
   8.23 MW at most, again on `L8-9-1`, the heaviest branch, the same pattern as step 3. Note that the grid is
   not pandapower's `case57` of step 2 (different dispatch, extra elements), so the two sets of flows are not
   comparable branch by branch; the test `test_toop57_toop.py` checks both relations under `make test-toop`.
5. **Octave + MATPOWER.** `[pf, Bbus, Bf] = matpower_dcpf('case57')` (Octave 11.3, MATPOWER 8.1) gives 80 flows
   whose first five are 97.8996, 94.8996, 58.4379, 13.8948 and 12.7802 MW, with $\sum|F| = 1919.4868$ MW and
   $\max|F| = 177.2260$ MW; `dc_vs_ac.main()` prints the same five numbers and the same two sums from `rundcpp`.
   pandapower's `case57` is converted from MATPOWER's, and both apply the tap ratio, so the two agree to the
   fourth decimal.

## Exercise 1

With $\Delta\theta = \theta_f - \theta_t$, the AC active flow of a lossless branch is
$p_f = |V_f||V_t|\sin(\Delta\theta)/x$ and the DC flow is $p_f^{DC} = \Delta\theta/x$. Their ratio does not
depend on $x$:

$$
\frac{p_f^{DC}}{p_f} = \frac{\Delta\theta}{|V_f||V_t|\sin\Delta\theta}, \qquad
\varepsilon = \frac{p_f^{DC} - p_f}{p_f} = \frac{1}{|V_f||V_t|}\cdot\frac{\Delta\theta}{\sin\Delta\theta} - 1 .
$$

Both factors are monotone on the given ranges: $\Delta\theta/\sin\Delta\theta$ grows from 1 at
$\Delta\theta \to 0$ to $1/\operatorname{sinc}(10^\circ) = 1/0.99493 = 1.0051$ at $10^\circ$, and
$1/(|V_f||V_t|)$ lies between $1/1.05^2 = 0.9070$ and $1/0.95^2 = 1.1080$. Hence

$$
-9.30\,\% = \frac{1}{1.05^2} - 1 \;\le\; \varepsilon \;\le\; \frac{1.0051}{0.95^2} - 1 = +11.37\,\% .
$$

The angle contributes only $+0.51$ % at $10^\circ$; the voltage magnitudes contribute about $\pm 10$ %, which is
why the DC approximation is a flat-voltage approximation before it is a small-angle one, and why a voltage
profile of 0.94–1.06 pu produces the errors of Lab step 3 even with $r = 0$. Measured against the DC flow
instead ($(p_f - p_f^{DC})/p_f^{DC} = |V_f||V_t|\operatorname{sinc}\Delta\theta - 1$) the bounds are
$-10.2$ % and $+10.25$ %. `dc_approximation.error_bounds` evaluates $\varepsilon$ on the four voltage corners
and 1,001 angles and returns the same two numbers.

```python title="dc_approximation.py"
--8<-- "docs/part-3-power-systems/ch16-dc-power-flow/src/dc_approximation.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch16-dc-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

$B_\text{bus} = A^\top\operatorname{diag}(b)A$ is a weighted Laplacian of a connected graph: symmetric, positive
semidefinite, with the one-dimensional null space spanned by $\mathbf 1$ (every row sums to zero), and its range
is the set of vectors orthogonal to $\mathbf 1$, that is, the balanced injections. For a balanced $P$ the
equation $B_\text{bus}\theta = P$ therefore has solutions, and they form the line $\theta^\star + c\,\mathbf 1$.
Choosing a slack $s$ and deleting its row and column picks the one solution with $\theta_s = 0$, which is
$\theta^{(s)} = \theta^\star - \theta^\star_s\,\mathbf 1$ (the reduced matrix is nonsingular, [Chapter 2](../../part-1-math/ch02-linear-algebra/index.md)).
Two slacks $s$ and $r$ give $\theta^{(r)} - \theta^{(s)} = (\theta^\star_s - \theta^\star_r)\,\mathbf 1$: the
angle vectors differ by a constant, so the angles depend on the slack. The flows do not, because
$F = \operatorname{diag}(b)A\theta$ and every row of $A$ has one $+1$ and one $-1$, so $A\,\mathbf 1 = 0$ and
$F^{(r)} - F^{(s)} = \operatorname{diag}(b)A\,c\,\mathbf 1 = 0$. The flow of a branch is a difference of two
angles, and the slack choice only adds the same constant to both. If the injections do not sum to zero there is
no solution of the full system; the reduced system then assigns the imbalance to the slack bus, $P_s =
-\sum_{i\ne s}P_i$, and the flows do depend on where the imbalance is absorbed ([Chapter 17](../ch17-sensitivity-factors-1/index.md)).

Numerically (`dc_approximation.main`), on the four-bus ring with $P = (0.8, 0.2, -0.5, -0.5)$ pu, slack 0 and
slack 3 give the same flows $(0.3333, 0.4667, 0.5333, -0.0333)$ pu while the angles move from
$(0, -1.910, -5.348, -4.966)^\circ$ to $(4.966, 3.056, -0.382, 0)^\circ$: a shift of $4.966^\circ$ on every bus.

## Exercise 3

`dc_sparse_solve.susceptance_matrices` assembles $A$ as a sparse `(20, 14)` incidence matrix from the branch
table of pandapower's `case14` (with $b = 1/(x\tau)$: three transformers have taps), then $B_f =
\operatorname{diag}(b)A$ and $B_\text{bus} = A^\top B_f$, 54 nonzeros. The slack row and column are dropped,
`splu` factorizes the `(13, 13)` matrix once, and `flows_many` solves all right-hand sides in one call:
`lu.solve(p[:, keep].T)` takes the `(13, 1000)` block and $F = (B_{f,\text{red}}\,\Theta)^\top$ has shape
`(1000, 20)`, the layout of ToOp's `unsplit_flow`. The snapshots are the base case plus Gaussian noise of 10 MW
per bus, rebalanced at the slack; snapshot 0 is the base case itself and matches `rundcpp` to
$7.5 \times 10^{-14}$ MW. The block solve and 1,000 separate `spsolve` calls agree exactly (difference 0.0). Time
(best of several runs): 0.53 ms for the factorization plus the block solve against 38–42 ms for the separate
solves, a factor of 70–80. The separate solves pay a symbolic and numeric factorization per call plus Python
overhead; the block solve pays one factorization and then one triangular sweep per snapshot in compiled code.
This is the PTDF idea in disguise: one topology, one factorization, many injection vectors.

```python title="dc_sparse_solve.py"
--8<-- "docs/part-3-power-systems/ch16-dc-power-flow/src/dc_sparse_solve.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch16-dc-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

`makebdc_shift('case14', 8, 3)` (Octave 11.3, MATPOWER 8.1) rebuilds $A$ from `F_BUS` and `T_BUS`, takes
$b = 1/(x\tau)$ with $\tau = 1$ where `TAP` is zero, and compares with `makeBdc`: the largest entry difference of
$B_\text{bus}$ is $0$. `Pfinj` of the unmodified case is zero in every entry (no branch has a `SHIFT`). After
setting `SHIFT = 3` on branch 8 (the transformer 4–7, $x = 0.2091$, $\tau = 0.978$), exactly one entry of `Pfinj`
is nonzero, $\text{Pfinj}_8 = -b_8 \cdot 3\pi/180 = -4.8897 \times 0.05236 = -0.25601$ pu ($-25.6$ MW), and
`Pbusinj` equals $A^\top\,\text{Pfinj}$ (an injection of $-0.256$ pu at bus 4 and $+0.256$ pu at bus 7): both
checks hold to $5.6 \times 10^{-17}$. This is the equivalent-injection formula of the worked example,
$P_{f,\text{shift}} = -\operatorname{diag}(b)\,\theta_\text{shift}$ and $P_\text{bus,shift} = A^\top P_{f,\text{shift}}$,
with MATPOWER's sign: a positive `SHIFT` reduces the flow from $f$ to $t$.

```octave title="makebdc_shift.m"
--8<-- "docs/part-3-power-systems/ch16-dc-power-flow/src/makebdc_shift.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch16-dc-power-flow/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

`psdf.ptdf_from_solves` builds the PTDF with the chapter's own tool, one DC power flow per column (a unit injected
at bus $i$, withdrawn at the slack), and `psdf_matrix` applies
$\text{PSDF}_{\ell k} = b_k(\delta_{\ell k} - H_{\ell f_k} + H_{\ell t_k})$. `flows_with_shift` is the independent
check: it models $\varphi_k$ as the injections $-b_k\varphi_k$ at $f_k$, $+b_k\varphi_k$ at $t_k$ in a plain DC
power flow and adds the direct term $b_k\varphi_k$ on branch $k$. On the four-bus ring ($x = 0.1, 0.2, 0.1,
0.2$, $P = (0.8, 0.2, -0.5, -0.5)$ pu) every PSDF entry is $\pm 1/\sum x = \pm 1.6667$ pu per radian: a shift
anywhere on a single loop drives the same circulating flow $\varphi/\sum_\text{loop} x$ around it. A $5^\circ$
shift on branch 0 moves the flows from $(0.3333, 0.4667, 0.5333, -0.0333)$ to $(0.4788, 0.3212, 0.6788,
-0.1788)$ pu, and the PSDF prediction agrees to $10^{-16}$ (the test repeats this for every branch). Scaled as
`compute_psdf` does, by `base_mva` and $\pi/180$, the column reads $\pm 2.909$ MW per degree.

The sign convention: ToOp's PSDF is derived for a branch flow $b_k(\theta_f - \theta_t + \varphi_k)$, so a
positive $\varphi_k$ *increases* the flow from $f$ to $t$. MATPOWER (and PowSyBl's tap changer `alpha`, which
follows the same transformer equation) define the shift the other way, $p_f = b(\theta_f - \theta_t -
\theta_\text{shift})$: a positive shift *decreases* the from-to flow (Exercise 4 showed $\text{Pfinj}_8 < 0$
for a positive `SHIFT`). If the grid model reports its angle in that convention, then $\varphi_k =
-\theta_\text{shift}$, which is exactly the minus sign that `get_shift_angles` applies in
`powsybl_backend.py` under its `TODO`. The minus would be unnecessary only if the model's angle already followed
the "positive shift increases the from-to flow" convention.

```python title="psdf.py"
--8<-- "docs/part-3-power-systems/ch16-dc-power-flow/src/psdf.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch16-dc-power-flow/src){ target="_blank" } to see or download all the files of this chapter.
