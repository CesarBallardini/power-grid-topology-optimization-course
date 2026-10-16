# Chapter 14 – Solutions

Solutions of the Lab and the Exercises of [Chapter 14](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`ybus_check.m`](src/ybus_check.m) (the worked example, Octave + MATPOWER)
and [`ybus_numpy.py`](src/ybus_numpy.py) (the same in NumPy against pandapower, Lab steps 1–4),
[`small_ybus.py`](src/small_ybus.py) (Exercises 1–2), [`bdc_check.m`](src/bdc_check.m) (Exercise 3, Octave),
[`kron_reduction.py`](src/kron_reduction.py) (Exercise 4) and [`long_line.py`](src/long_line.py) (Exercise 5).
Each Python module's `main()` prints every number quoted below; the Octave numbers come from GNU Octave 11.3 with
MATPOWER 8.1 (pandapower 3.5 for the Python path). Where the two paths overlap they agree to round-off.

## Lab

1. **Ybus from the tables.** `ybus_numpy.case14_ppc()` loads `pandapower.networks.case14()`, runs one power flow
   and returns `net._ppc`: 14 buses, 20 branches, `baseMVA = 100`. `connection_matrices` builds $C_f$ and $C_t$
   (20 × 14, one 1 per row, from the `F_BUS` and `T_BUS` columns, already 0-based in the `ppc`), `branch_admittances`
   the four entries of every branch from `BR_R`, `BR_X`, `BR_B`, `TAP` and `SHIFT`,

    $$
    y_{tt} = y_s + j\tfrac{b_c}{2}, \quad y_{ff} = \frac{y_{tt}}{\tau^2}, \quad
    y_{ft} = -\frac{y_s}{\tau e^{-j\theta}}, \quad y_{tf} = -\frac{y_s}{\tau e^{j\theta}},
    $$

    and `build_ybus` assembles $Y_f = [Y_{ff}] C_f + [Y_{ft}] C_t$, $Y_t = [Y_{tf}] C_f + [Y_{tt}] C_t$ and
    $Y_\text{bus} = C_f^\top Y_f + C_t^\top Y_t + [Y_\text{sh}]$ with $Y_\text{sh} = (G_s + jB_s)/\text{baseMVA}$
    from the bus table (only bus 9 has a shunt, 19 Mvar). The Octave path is `ybus_check.m` unchanged.

2. **Comparison.** `max_difference(ppc)` gives $\max |Y_\text{bus} - Y_\text{bus}^\text{makeYbus}| = 0$ exactly: both
   codes do the same floating-point operations in the same order. Octave: `ybus_check('case14')` returns
   $1.8 \times 10^{-15}$ and `ybus_check('case57')` $1.4 \times 10^{-14}$, round-off from the sparse products.

3. **Sparsity, taps, symmetry.** `matplotlib.pyplot.spy(Ybus)` shows 54 nonzeros of 196 (28 %): the 14 diagonal
   entries and two per branch (figure described, not saved; with 20 branches and 14 buses the pattern is the
   adjacency of the IEEE 14 graph plus the diagonal). The transformers are the branches with `TAP` different
   from 1: branches 15, 16, 17 of the `ppc` (0-based), i.e. 4–7 with $\tau = 0.978$, 4–9 with $\tau = 0.969$ and
   5–6 with $\tau = 0.932$ (1-based bus numbers), none with a phase shift. $\lVert Y_\text{bus} - Y_\text{bus}^\top
   \rVert_F = 0$: an off-nominal tap scales $y_{ff}$ by $1/\tau^2$ and both off-diagonals by $1/\tau$, so the matrix
   stays symmetric; only a phase shift breaks the symmetry.

4. **A 5° phase shift.** `with_shift(branch, 15, 5.0)` puts $\theta = 5°$ on the 4–7 transformer. Rebuilt,
   $\lVert Y_\text{bus} - Y_\text{bus}^\top \rVert_F = 1.2053$, coming from the single pair
   $Y_{4,7} = -0.4261 + 4.8709j$ and $Y_{7,4} = +0.4261 + 4.8709j$: the two entries have the same modulus 4.8895
   ($= 1/(x\tau)$) and opposite real parts, since $y_{ft} = -y_s e^{j\theta}/\tau$ and $y_{tf} = -y_s
   e^{-j\theta}/\tau$ are complex conjugates up to the common factor $-y_s/\tau$. The norm is
   $\sqrt{2}\,|Y_{4,7} - Y_{7,4}| = \sqrt{2} \times 2 \times 0.4261 = 1.2053$, i.e. $2\sqrt{2}\,\sin 5° / (x \tau)$.

    ```python title="ybus_numpy.py"
    --8<-- "docs/part-3-power-systems/ch14-component-models/src/ybus_numpy.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch14-component-models/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Series admittances: $1/j0.1 = -j10$ (line 1–2), $1/j0.2 = -j5$ (line 2–3), $1/j0.25 = -j4$ (line 1–3). Each line
adds $y_s + jb_c/2 = y_s + j0.01$ to the diagonal entry of both its buses and $-y_s$ to the two off-diagonal
entries between them. Bus 1 has lines 1–2 and 1–3, bus 2 has 1–2 and 2–3, bus 3 has 2–3 and 1–3:

$$
Y_\text{bus} = \begin{bmatrix}
-j13.98 & j10 & j4 \\
j10 & -j14.98 & j5 \\
j4 & j5 & -j8.98
\end{bmatrix},
$$

for example $Y_{11} = -j10 - j4 + 2 \times j0.01 = -j13.98$. Every row sums to $j0.02$: the series admittances cancel
between the diagonal and the off-diagonal entries of the same row, and what remains is the total charging
connected to the bus, two half-lines of $j0.01$ each. In general $\sum_k Y_{ik}$ is the total shunt admittance at
bus $i$ (charging plus explicit shunts), zero for a lossless network without charging, which is why
$B_\text{bus}$ of the DC power flow is singular. `small_ybus.ring3_ybus()` and `row_sums` reproduce the matrix and
the row sums of $j0.02$.

## Exercise 2

With $y_s = 1/(j0.1) = -j10$, $\tau = 1.05$ and $\theta = 0$,

$$
Y_\text{br} = \begin{bmatrix} y_s/\tau^2 & -y_s/\tau \\ -y_s/\tau & y_s \end{bmatrix}
= \begin{bmatrix} -j9.0703 & j9.5238 \\ j9.5238 & -j10 \end{bmatrix}.
$$

A $\pi$ circuit with series admittance $y_\text{se}$ and shunts $y_1$, $y_2$ has
$Y = \begin{bmatrix} y_\text{se} + y_1 & -y_\text{se} \\ -y_\text{se} & y_\text{se} + y_2 \end{bmatrix}$, so matching
the entries:

$$
y_\text{se} = \frac{y_s}{\tau} = -j9.5238, \qquad
y_1 = y_s\left(\frac{1}{\tau^2} - \frac{1}{\tau}\right) = +j0.4535, \qquad
y_2 = y_s\left(1 - \frac{1}{\tau}\right) = -j0.4762 .
$$

The shunt at the tapped (from) side is capacitive ($+j$) and the one at the other side inductive ($-j$); they
nearly cancel ($y_1 + y_2 = y_s (1/\tau - 1)^2 = -j0.0227$). The circuit is asymmetric because the ideal transformer
sits at one end: seen from bus 1 the reactance is scaled by $\tau^2$, seen from bus 2 it is not, so the two
driving-point admittances $y_s/\tau^2$ and $y_s$ differ while the transfer admittance is the geometric mean
$y_s/\tau$ of the two. A symmetric $\pi$ would need $y_1 = y_2$, i.e. $\tau = 1$. The "negative" shunt is a
mathematical artefact of forcing a transformer into a $\pi$ template, not a physical capacitor; it is nevertheless
what every power-flow program stores. `small_ybus.transformer_ybr(1.05, 0.1)` and `pi_equivalent` print these
values.

```python title="small_ybus.py"
--8<-- "docs/part-3-power-systems/ch14-component-models/src/small_ybus.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch14-component-models/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 3

`bdc_check.m` builds $A = C_f - C_t$, $B_\text{plain} = A^\top \operatorname{diag}(1/x)\, A$ and subtracts it from
the first output of `makeBdc(mpc)`. For `case14` the difference is zero everywhere except at the buses of the
three transformers, where MATPOWER uses $b = 1/(x\tau)$ instead of $1/x$ (`makeBdc`: `b = b ./ tap`):

| branch | $\tau$ | $x$ | $1/x$ | $1/(x\tau)$ | difference |
|---|---|---|---|---|---|
| 4–7 | 0.978 | 0.20912 | 4.7819 | 4.8895 | 0.10757 |
| 4–9 | 0.969 | 0.55618 | 1.7980 | 1.8555 | 0.05752 |
| 5–6 | 0.932 | 0.25202 | 3.9679 | 4.2574 | 0.28951 |

The eleven entries that differ (1-based) are exactly the ones these branches touch: $(5,5)$ and $(6,6)$ by
$+0.28951$, $(5,6)$ and $(6,5)$ by $-0.28951$; $(4,7)$ and $(7,4)$ by $-0.10757$, $(7,7)$ by $+0.10757$; $(4,9)$ and
$(9,4)$ by $-0.05752$, $(9,9)$ by $+0.05752$; and $(4,4)$ by $+0.10757 + 0.05752 = 0.16509$, the sum of its two
transformers. The largest difference is 0.28951 (the 5–6 transformer, with the smallest $\tau$), and for `case9`,
which has no taps, the two matrices are identical ($0$). Nothing else differs: phase shifts, if there were any,
would not appear in $B_\text{bus}$ at all but in the injection vectors `Pbusinj` and `Pfinj` (the third and fourth
outputs; here zero), because in the DC model a shift adds a constant to the angle difference rather than a term
to the matrix. The lesson for `get_susceptance_matrices` in ToOp: $b = 1/x$ is the DC model of a line, and the
tap ratio of a transformer enters the DC susceptance as $1/(x\tau)$, a correction of a few percent that
MATPOWER applies and a plain Laplacian forgets.

```octave title="bdc_check.m"
--8<-- "docs/part-3-power-systems/ch14-component-models/src/bdc_check.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch14-component-models/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 4

Partition the nodal equations $I = Y V$ into kept ($k$) and eliminated ($e$) buses:

$$
\begin{bmatrix} I_k \\ 0 \end{bmatrix} =
\begin{bmatrix} Y_{kk} & Y_{ke} \\ Y_{ek} & Y_{ee} \end{bmatrix}
\begin{bmatrix} V_k \\ V_e \end{bmatrix}
\;\Rightarrow\; V_e = -Y_{ee}^{-1} Y_{ek} V_k, \qquad
I_k = \left(Y_{kk} - Y_{ke} Y_{ee}^{-1} Y_{ek}\right) V_k .
$$

The Schur complement $Y_\text{red} = Y_{kk} - Y_{ke} Y_{ee}^{-1} Y_{ek}$ is the Ward-type equivalent: an admittance
matrix on the kept buses alone that gives the same $V_k$ for every injection with $I_e = 0$. `kron_reduction.KEEP`
keeps buses 1–5 of IEEE 14 (the 132 kV part) and eliminates 6–14. Checks (`kron_reduction.main()`): with a
deterministic pseudo-random complex injection at the kept buses, $\max |V_k^\text{full} - V_k^\text{reduced}| =
3.5 \times 10^{-13}$, round-off; the reduced matrix is symmetric (the original is). Only buses 4 and 5 are
neighbours of the eliminated set (through the transformers 4–7, 4–9 and 5–6), so the reduction changes only the
block $\{4, 5\}$: $Y_{4,5}$ goes from $-6.841 + 21.579j$ to $-7.101 + 23.060j$, and the diagonals of 4 and 5 move
with it, while the entries of buses 1–3 are untouched. In general the elimination creates a fictitious branch
between every pair of kept buses that share an eliminated neighbour (fill-in), which is why equivalents of a large
external network are dense. The equivalent is only valid for the injection pattern it assumes: adding 1 p.u. of
current at the eliminated bus 9 moves the kept voltages by 2.47 p.u. relative to the reduced solution. A Ward
equivalent therefore carries, besides $Y_\text{red}$, equivalent injections at the boundary buses that stand for the
eliminated loads and generators, and an *extended* Ward adds a fictitious generator to hold the boundary voltage.

```python title="kron_reduction.py"
--8<-- "docs/part-3-power-systems/ch14-component-models/src/kron_reduction.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch14-component-models/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

The statement leaves the per-km parameters to you; `long_line.py` uses a typical single-circuit 400 kV line at 50
Hz: $r = 0.028\ \Omega/\text{km}$, $L = 0.9$ mH/km ($x = 0.2827\ \Omega/\text{km}$), $C = 12.5$ nF/km
($b = 3.927\ \mu\text{S/km}$), $g = 0$. Then $\gamma = \sqrt{zy} = 5.21 \times 10^{-5} + j1.055 \times 10^{-3}$
per km (wavelength $2\pi/\beta = 5956$ km) and $Z_c = \sqrt{z/y} = 268.7 - j13.3\ \Omega$. For a length $l$,

$$
A = D = \cosh \gamma l, \qquad B = Z_c \sinh \gamma l, \qquad C = \frac{\sinh \gamma l}{Z_c},
\qquad AD - BC = 1 ,
$$

and at 300 km $A = 0.9504 + 0.0049j$, $B = 8.12 + 83.43j\ \Omega$, $C = -1.9 \times 10^{-6} + 1.159 \times 10^{-3}j$ S.
The exact equivalent $\pi$ has $Z' = B = Z \,\sinh(\gamma l)/(\gamma l)$ and $Y'/2 = (A - 1)/B = (Y/2)
\tanh(\gamma l/2)/(\gamma l/2)$, against the nominal $Z = zl = 8.40 + 84.82j\ \Omega$ and $Y = yl = 1.178 \times
10^{-3}j$ S: the exact series impedance is 1.7 % smaller and the shunt 0.8 % larger. The relative error of the
nominal series admittance $1/Z$ is $|\gamma l / \sinh(\gamma l) - 1| \approx |\gamma l|^2/6$:

| length (km) | 50 | 100 | 200 | 300 | 400 | 500 |
|---|---|---|---|---|---|---|
| error of $1/Z$ (%) | 0.05 | 0.19 | 0.75 | 1.69 | 3.04 | 4.80 |

The plot (described, not saved) is a parabola: the error grows with the square of the length, 1.67 % from the
rule of thumb at 300 km against 1.69 % exact, and crosses 1 % at about 230 km. That is the usual textbook
boundary between "medium" lines, where the nominal $\pi$ is adequate, and "long" lines, where the exact
$\pi$ (or the ABCD form) is needed. For the branch tables of Chapter 14 the practical consequence is that the $r$,
$x$, $b_c$ of a long line in a MATPOWER or pandapower case should already be the *equivalent* $\pi$ values, which is
what grid data from TSOs normally contains.

```python title="long_line.py"
--8<-- "docs/part-3-power-systems/ch14-component-models/src/long_line.py"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-3-power-systems/ch14-component-models/src){ target="_blank" } to see or download all the files of this chapter.
