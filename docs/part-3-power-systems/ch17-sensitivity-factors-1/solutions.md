# Chapter 17 – Solutions

Solutions of the Lab and the Exercises of [Chapter 17](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`lodf.py`](src/lodf.py) (the worked example: PTDF, LODF column, N-1 flows),
[`n1_screening.py`](src/n1_screening.py) (the Lab and Exercises 3, 5 and 6 on IEEE 14 and 118),
[`triangle.py`](src/triangle.py) (Exercise 1), [`matpower_lodf.m`](src/matpower_lodf.m) and
[`ptdf_slack_case30.m`](src/ptdf_slack_case30.m) (Lab step 3 and Exercise 4 in Octave + MATPOWER) and
[`test_n1_screening_toop.py`](src/test_n1_screening_toop.py) (Lab step 2 against ToOp's reference, runs under
`make test-toop`). Every `main()` prints the numbers quoted below; the ToOp comparison used ToOp `ba1874e`, the
timings are from one laptop. Branch and bus indices are pandapower's, 0-based, lines first and transformers last,
so they differ from MATPOWER's. The Lab steps are also in [lab.ipynb](src/lab.ipynb) and the exercise statements in
[exercises.ipynb](src/exercises.ipynb).

## Lab

1. **PTDF, PSDF and LODF for IEEE 14 and 118.** `n1_screening.case_data` reads pandapower's branch table after
   `rundcpp` (from, to, $x\tau$, balanced injections, slack: bus 0 for `case14`, bus 68 for `case118`).
   `lodf.ptdf_matrix` gives $H$ of shape $(20, 14)$ and $(186, 118)$ with a zero slack column;
   `n1_screening.psdf_matrix` gives $\text{PSDF} = (I - H_{:,f} + H_{:,t})/x$ of shape $(20, 20)$ and
   $(186, 186)$ (pu per radian; `test_n1_screening.py` checks a column against a 3° shift modelled as equivalent
   injections); `lodf_matrix` builds the whole LODF from $\mathcal H = H(C_f - C_t)^\top$ in one expression,
   $\text{LODF} = \mathcal H / (1 - \operatorname{diag}\mathcal H)$ with $-1$ on the diagonal and NaN columns where
   the denominator is below $10^{-11}$. The largest base-case flows are 147.8 MW (IEEE 14) and 450.0 MW (IEEE
   118).

    ```python title="n1_screening.py"
    --8<-- "docs/part-3-power-systems/ch17-sensitivity-factors-1/src/n1_screening.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch17-sensitivity-factors-1/src){ target="_blank" } to see or download all the files of this chapter.

2. **Validation.** `n1_matrix(lodf, f0)` is `f0[None, :] + lodf.T * f0[:, None]`, the whole N-1 table of shape
   `(n_outages, n_branches)`. Against `brute_force_n1`, one DC power flow per removed branch, the largest
   difference is $6.5 \times 10^{-13}$ MW on the 19 non-islanding outages of IEEE 14 and $2.7 \times 10^{-12}$
   MW on the 177 of IEEE 118: the LODF update is exact, not approximate, because both are the same linear
   model. Against ToOp's `numpy_reference.contingency_analysis(ptdf, from_node, to_node, branches_to_outage,
   nodal_injections)` the N-1 rows agree to $10^{-9}$ MW and the `success` flags are `False` exactly for our NaN
   columns; `calc_lodf` matches `lodf_matrix` column by column to $10^{-12}$ (`test_n1_screening_toop.py`, two
   tests passed with ToOp's interpreter). ToOp's reference uses the same $10^{-11}$ threshold on the denominator.
3. **Bridges.** IEEE 14 has one bridge, branch 18 (buses 6–7, the transformer 7–8 of the one-line diagram that
   feeds bus 8 alone): $1 - h_{kk} = 0$ to the last digit, and the next smallest denominator is 0.1113, so there
   is no ambiguity. IEEE 118 has nine: branches 6 (7–8), 7 (8–9), 103 (70–72), 121 (84–85), 163 (109–110), 164
   (109–111), 170 (11–116), 184 (86–85) and 185 (115–67) in pandapower's numbering (the leaves 8–9–10, 73, 86–87,
   111, 112, 116, 117 and 68 of the one-line diagram); all have $1 - h_{kk}$ at round-off level, and the smallest
   denominator of a non-bridge is 0.0531. `lodf_matrix` returns NaN in exactly those columns, and `brute_force_n1`
   cannot solve the islanded grid for the same outages. The libraries are less careful:
    - pandapower's `pandapower.pypower.makeLODF` agrees with ours to $10^{-14}$ on every non-bridge column and
      flags the IEEE 14 bridge (infinite column); on IEEE 118 it flags only 7 of the 9. For branches 6 and 170
      its denominator is $1.1 \times 10^{-16}$ instead of 0, so the column is round-off divided by round-off:
      finite, plausible-looking factors between $-2.4$ and $6.0$, and no warning. A screening code that trusts
      them would report flows on an islanded grid.
    - MATPOWER 8.1 (`matpower_lodf`, Octave 11.3): `makeLODF(mpc, H, true)` on `case14` masks the bridge
      (NaN column 14 in MATPOWER's branch order) and the LODF from a distributed-slack PTDF equals the
      single-slack one to $4.4 \times 10^{-16}$. On `case118` the plain `makeLODF(mpc, H)` returns non-finite
      columns for only 2 of the 9 bridges (columns 7 and 184), and `makeLODF(mpc, H, true)` fails inside
      `find_bridges` with `br(197): out of bound 186`: the bridge finder trips over the parallel branches of
      `case118` (a bug in MATPOWER 8.1's `find_bridges`). Find the bridges with the graph
      ([Chapter 3](../../part-1-math/ch03-graph-theory/index.md)) and mask the columns yourself.

    ```octave title="matpower_lodf.m"
    --8<-- "docs/part-3-power-systems/ch17-sensitivity-factors-1/src/matpower_lodf.m"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch17-sensitivity-factors-1/src){ target="_blank" } to see or download all the files of this chapter.

4. **Distributed slack.** With uniform weights $w = \mathbf 1/n$, `distributed_slack_ptdf` applies
   $H_w = H(I - w\mathbf 1^\top)$: the PTDF entries move by up to 0.618 (IEEE 14) and 0.444 (IEEE 118), but the
   base-case flows change by $8.5 \times 10^{-14}$ and $1.7 \times 10^{-12}$ MW and the LODF by $4.4 \times
   10^{-16}$ and $1.7 \times 10^{-15}$: nothing. Losing 40 MW at the largest non-slack generator (bus 1 of
   IEEE 14, bus 88 of IEEE 118) gives post-outage flows that differ by up to 24.7 MW and 17.8 MW between the two
   slacks, because the single slack puts the whole 40 MW back at one bus while the distributed slack spreads it
   over every bus.
5. **PTDF update.** `ptdf_after_outage(h, frm, to, k)` applies $H' = H + \text{LODF}_{:,k}H_{k,:}$ and zeroes row
   $k$. Against `ptdf_matrix` on the branch table without branch $k$ the largest difference is
   $3.3 \times 10^{-15}$ (IEEE 14) and $3.2 \times 10^{-15}$ (IEEE 118); the test repeats it for three branches.
   This rank-one update is the first step of the disconnection and bus-split factors of
   [Chapter 18](../ch18-sensitivity-factors-2/index.md).

## Exercise 1

Buses 1, 2, 3, branches 1–2, 1–3, 2–3, $x = 1$ each, slack at bus 3. Remove row and column 3 of
$B_\text{bus} = \begin{bmatrix} 2 & -1 & -1 \\ -1 & 2 & -1 \\ -1 & -1 & 2 \end{bmatrix}$ to get
$\tilde B = \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}$, $\tilde B^{-1} = \frac13\begin{bmatrix} 2 & 1 \\ 1 & 2
\end{bmatrix}$. One MW at bus 1 (withdrawn at bus 3) gives $\theta = (2/3, 1/3, 0)$, hence flows
$1/3$ on 1–2, $2/3$ on 1–3, $1/3$ on 2–3: the direct path has reactance 1, the detour 1–2–3 has 2, and the
current divider sends $2/3$ and $1/3$. One MW at bus 2 gives $\theta = (1/3, 2/3, 0)$ and flows $-1/3$, $1/3$,
$2/3$. So

$$
H = \begin{bmatrix} 1/3 & -1/3 & 0 \\ 2/3 & 1/3 & 0 \\ 1/3 & 2/3 & 0 \end{bmatrix}
\quad\text{(rows 1–2, 1–3, 2–3; columns bus 1, 2, 3).}
$$

For the LODF, $h_{\ell k} = H_{\ell f_k} - H_{\ell t_k}$. Every $h_{kk} = 2/3$: sending one MW from $f_k$ to $t_k$
with branch $k$ in service, $2/3$ take the branch and $1/3$ the two others in series (reactance 2 against 1).
Hence $1 - h_{kk} = 1/3$ and $\text{LODF}_{\ell k} = h_{\ell k}/(1/3) = 3h_{\ell k}$ with $h_{\ell k} = \pm 1/3$ off
the diagonal:

$$
\text{LODF} = \begin{bmatrix} -1 & 1 & -1 \\ 1 & -1 & 1 \\ -1 & 1 & -1 \end{bmatrix}
\quad\text{(rows: monitored branch; columns: outaged branch).}
$$

Current divider check: after the outage of 1–2 the power it carried has exactly one path left, 1–3 then 3–2 in
series, so all of it appears on 1–3 ($\text{LODF} = +1$) and on 2–3 with the opposite orientation
($\text{LODF} = -1$); the signs follow the from–to orientation of each branch. `triangle.main()` prints both
matrices and checks the base flows for 1 MW from bus 1 to bus 3, $(1/3, 2/3, 1/3)$, becoming $(0, 1, 0)$ after
the outage of 1–2.

```python title="triangle.py"
--8<-- "docs/part-3-power-systems/ch17-sensitivity-factors-1/src/triangle.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch17-sensitivity-factors-1/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

Let $k$ join $f$ and $t$ with susceptance $b_k$, and let $\tilde B$ be the grounded Laplacian (slack row and
column removed; it is nonsingular because the grid is connected). $h_{kk}$ is the flow on $k$ when one unit is
injected at $f$ and withdrawn at $t$: with $\theta = \tilde B^{-1}(e_f - e_t)$ (slack entry 0),
$h_{kk} = b_k(\theta_f - \theta_t)$. Write $\tilde B = \tilde B_{-k} + b_k(e_f - e_t)(e_f - e_t)^\top$, where
$\tilde B_{-k}$ is the grounded Laplacian of the grid without branch $k$.

*If $k$ is a bridge*, removing it disconnects $f$ from $t$: in the grid without $k$ no power can go from $f$ to
$t$, so the unit injected at $f$ must cross branch $k$ entirely, $h_{kk} = 1$. Algebraically: the component
containing $t$ (say) has no slack, so $\tilde B_{-k}$ restricted to it has the null vector $\mathbf 1_T$
(indicator of that component), and $(e_f - e_t)^\top\mathbf 1_T = -1 \ne 0$; from $\tilde B\theta = e_f - e_t$
and $\mathbf 1_T^\top\tilde B_{-k} = 0$ one gets $\mathbf 1_T^\top(e_f - e_t) = b_k(\theta_f - \theta_t)\,
\mathbf 1_T^\top(e_f - e_t)$, i.e. $h_{kk} = b_k(\theta_f - \theta_t) = 1$.

*If $k$ is not a bridge*, the grid without $k$ is still connected, so $\tilde B_{-k}$ is a grounded Laplacian of a
connected graph and therefore positive definite ([Chapter 2](../../part-1-math/ch02-linear-algebra/index.md)). Multiply
$\tilde B\theta = e_f - e_t$ on the left by $\theta^\top$: $\theta^\top\tilde B_{-k}\theta + b_k(\theta_f -
\theta_t)^2 = \theta_f - \theta_t$. Put $d = \theta_f - \theta_t$ (nonzero, since $\theta^\top\tilde B\theta > 0$):
$\theta^\top\tilde B_{-k}\theta = d - b_kd^2 = d(1 - h_{kk})$. The left side is strictly positive, and $d > 0$
(it equals $(e_f - e_t)^\top\tilde B^{-1}(e_f - e_t)$ with $\tilde B^{-1}$ positive definite), so $1 - h_{kk} > 0$,
i.e. $h_{kk} < 1$. Together: $h_{kk} = 1$ if and only if $k$ is a bridge, and otherwise $0 < h_{kk} < 1$
(the lower bound because $h_{kk} = b_k d$ with $d > 0$). This is why ToOp's `calc_lodf` can use
$|1 - h_{kk}| \le 10^{-11}$ as its islanding test: there is no non-bridge with a denominator near zero, only
the round-off of true bridges. The numbers of Lab step 3 (0 on every bridge, at least 0.0531 elsewhere on IEEE
118) illustrate the gap.

## Exercise 3

`n1_matrix(lodf_matrix(h, frm, to), h @ p)` is the one broadcasted expression: `lodf_matrix` forms
$\mathcal H = H_{:,f} - H_{:,t}$ (186 × 186), divides by $1 - \operatorname{diag}\mathcal H$ and sets the
diagonal; `n1_matrix` returns `f0[None, :] + lodf.T * f0[:, None]`, shape `(186, 186)`, row $k$ holding the flows
after the outage of $k$. On IEEE 118 the whole screen takes 0.12–0.21 ms (best of five, including the LODF
build), against 84–98 ms for 186 separate DC power flows (`brute_force_n1`: a connectivity check and one
`scipy.linalg.solve` of the 117 × 117 reduced system per outage), a factor of about 400–700. The brute force is
itself fast only because the matrices are dense and small; its cost is one factorization per outage, the LODF's
is one factorization for all of them plus a rank-one update per outage. Both give the same flows to
$2.7 \times 10^{-12}$ MW on the 177 non-islanding outages (Lab step 2), and the nine bridges are NaN rows.

A note on the measurement: `numpy.linalg.solve` with numpy's bundled multithreaded OpenBLAS took 8.9 ms per
117 × 117 solve on this laptop (0.07 ms with `OPENBLAS_NUM_THREADS=1`), which would have made the brute force look
120 times slower than it is; `brute_force_n1` therefore uses `scipy.linalg.solve(..., assume_a='pos')`, 0.15 ms
per solve. Check your BLAS before believing a benchmark.

## Exercise 4

`ptdf_slack_case30('case30')` (Octave 11.3, MATPOWER 8.1) computes `H = makePTDF(mpc)` (single slack, the
reference bus 1) and `Hw = makePTDF(mpc, w)` with $w$ proportional to the generator outputs: buses 1, 2, 22, 27,
23 and 13 with 23.5, 61.0, 21.6, 26.9, 19.2 and 37.0 MW. The PTDF is 41 × 30. The difference `Hw - H` has
**identical columns**: every column changes by the same vector $-Hw$, as $H_w = H(I - w\mathbf 1^\top) = H -
(Hw)\mathbf 1^\top$ says. So no column changes "more" than another; the largest entry of the common change is
0.6296, on branch 1 (buses 1–2), the line next to the former slack: with a single slack, 63 % of any MW injected
elsewhere came back through 1–2 to bus 1, and with the slack spread over six generators that return flow is
shared. The column of bus 1 is the one that changes *qualitatively*: it was identically zero (injecting at the
slack and withdrawing at the slack does nothing) and becomes $-Hw$, with entries up to 0.6296. The base-case
flows do not change: for a balanced injection vector $P$ ($\mathbf 1^\top P = 0$), $H_wP = HP - (Hw)(\mathbf
1^\top P) = HP$; the Octave check (`makeSbus` injections, with the imbalance of the losses removed with the same
weights) gives a change of $1.1 \times 10^{-14}$ MW. The slack only matters for where an *unbalanced* change is
absorbed.

```octave title="ptdf_slack_case30.m"
--8<-- "docs/part-3-power-systems/ch17-sensitivity-factors-1/src/ptdf_slack_case30.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch17-sensitivity-factors-1/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

`generator_outage(h, p, n, mw)` returns $F = H_w(P - \text{mw}\,e_n)$ with $H_w = H$ (single slack) or
$H(I - w\mathbf 1^\top)$ (distributed). The difference between the two is
$\text{mw}\,(H_w - H)_{:,n} = -\text{mw}\,(H e_n - Hw)\ldots$ more simply: $\Delta F = \text{mw}\,(Hw - H_{:,s})$
where $H_{:,s} = 0$, i.e. $\Delta F = \text{mw}\,Hw$, the flow pattern of moving `mw` from the slack bus to the
weighted buses. It does not depend on $n$ at all: whatever generator is lost, the two models differ by the same
redistribution of the pick-up. On IEEE 14, losing 40 MW at bus 1 (the 40 MW unit at bus 2 of the diagram) with
uniform weights gives differences up to 24.7 MW per branch, concentrated on the branches leaving the slack bus
(lines 1–2 and 1–5), which under a single slack carry all of the replacement power; on IEEE 118, 40 MW lost at
bus 88 gives up to 17.8 MW. `examples.ipynb` draws the bar chart (branch index against `single - shared` in
MW): two tall bars at the slack's branches, a decaying tail elsewhere, and zeros on branches far from both the
slack and the participating generators.

The link to ToOp: its `calc_injection_outage` adds $H_{:,n}\Delta P$ with a single-slack PTDF, so after a
generator outage it predicts that the slack bus picks up the loss. The AC references (and the real system) share
it among the generators, $H_w$-style. The post-outage AC–DC mismatch therefore contains, on top of the base-case
mismatch $m$, a term $\text{mw}\,Hw$ that the constant-offset correction of the worked example does not know:
exactly the bars of the plot. For line outages this term is absent (the LODF is slack independent), which is
why the chapter says the single slack is harmless for N-1 branch screening and not for injection outages.

## Exercise 6

With $\alpha = 1$ the base case is $F^{(0)} = HP + m$, $m = F^{AC} - F^{DC}$. Every later topology acts
linearly on $F^{(0)}$. After the outage of $k_1$:

$$
F^{(1)}_\ell = F^{(0)}_\ell + \text{LODF}_{\ell k_1}F^{(0)}_{k_1}
= \underbrace{(HP)_\ell + \text{LODF}_{\ell k_1}(HP)_{k_1}}_{\text{DC flow without } k_1}
+ \underbrace{m_\ell + \text{LODF}_{\ell k_1}m_{k_1}}_{\text{propagated offset}} .
$$

The second outage uses the LODF of the grid without $k_1$, $\text{LODF}'$, computed from the updated PTDF
$H' = H + \text{LODF}_{:,k_1}H_{k_1,:}$ (Lab step 5). Applying it to $F^{(1)}$:

$$
F^{(1,2)}_\ell = F^{(1)}_\ell + \text{LODF}'_{\ell k_2}F^{(1)}_{k_2}, \qquad\text{so}\qquad
F^{(1,2)}_\ell = \bigl[\text{DC flow without } k_1, k_2\bigr]_\ell
+ m_\ell + \text{LODF}_{\ell k_1}m_{k_1} + \text{LODF}'_{\ell k_2}\bigl(m_{k_2} + \text{LODF}_{k_2 k_1}m_{k_1}\bigr).
$$

The DC part is exact: `double_outage` applies the two steps on IEEE 14 (branches 2 and 9) and matches a DC power
flow without both branches to $8.5 \times 10^{-14}$ MW, in either order. The correction part is the base-case
mismatch pushed through the two LODFs as if it were power, which is what "constant offset" means: the AC–DC
gap of each branch is treated as a fixed injection pattern that the outages redistribute like any other flow,
never recomputed from an AC solution of the new topology.

When is that acceptable? The offset $m$ is made of losses and reactive-flow effects of the base case, which are
roughly quadratic in the branch loadings and tied to the base-case voltage profile. The assumption holds when the
outages move little flow relative to the base case (small LODFs, lightly loaded corridors), when the voltage
profile is held by nearby generators so that the AC solution of the new topology stays close to the old one,
and for a single or double outage far from the slack and from the lost branches' corridors. It fails when an
outage reroutes a large share of a corridor's power (the losses on the receiving lines grow with the square of
the flow, and $m$ was computed for the old flow), when the post-contingency case has low voltages or reactive
problems the DC model cannot see ([Chapter 15](../ch15-ac-power-flow/index.md), Exercise 5), when an outage is a
generator (Exercise 5 adds a slack term), and of course for the bus splits of Chapter 18, where the base-case
AC solution no longer describes the station. This is why ToOp uses the correction for screening and sends the
candidates to a full AC N-1 anyway ([Chapter 19](../ch19-operational-security/index.md)).
