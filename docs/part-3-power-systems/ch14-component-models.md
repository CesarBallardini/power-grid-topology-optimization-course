# Chapter 14. Network component models and network matrices

**Weeks:** 2 · **Semester:** 3 · **Prerequisites:** [Chapter 11](../part-2-circuits/ch11-transformers-psts-hvdc.md), [Chapter 12](ch12-power-grid-operation.md)

!!! info "Why it matters for ToOp"

    The two backends turn grid elements into branch susceptances and injections
    (`dc_solver/preprocess/pandapower/pandapower_backend.py`, `dc_solver/preprocess/powsybl/powsybl_backend.py`):
    lines, two- and three-winding transformers, PSTs, Ward and extended Ward equivalents, dangling (boundary)
    lines, tie lines ($x = x_1 + x_2$), HVDC as injections. The matrices built here reappear as
    $B_\text{bus} = A^\top \operatorname{diag}(b)\, A$ in `get_susceptance_matrices`
    (`dc_solver/preprocess/helpers/ptdf.py`), the starting point of every DC computation in ToOp.

## Topics

- Transmission line parameters; short, medium ($\pi$) and long line models.
- Transformer models with off-nominal tap and phase shift in the branch admittance matrix.
- Generators, loads (constant P, constant Z), shunts.
- **Bus admittance matrix $Y_\text{bus}$**: construction by inspection and via the connection matrices
  $C_f, C_t$ (revisits the incidence matrix and Laplacian of
  [Chapter 2](../part-1-math/ch02-linear-algebra.md) and [Chapter 3](../part-1-math/ch03-graph-theory.md) with
  complex weights); sparsity; loss of symmetry with phase shifters.
- Network equivalents: Ward / extended Ward (Kron reduction as a Schur complement), boundary nodes (X-nodes),
  tie lines.
- Tools: MATPOWER in GNU Octave (`loadcase`, `ext2int`, `makeYbus`, `define_constants`); pandapower's internal
  `ppc` and its PYPOWER port of `makeYbus`.

## Learning outcomes

- Build $Y_\text{bus}$ of a small network by inspection and with $C_f^\top Y_f + C_t^\top Y_t + [Y_\text{sh}]$,
  including off-nominal taps and phase shifters.
- Reproduce MATPOWER's `makeYbus` (Octave) or pandapower's `makeYbus` (NumPy) for IEEE 14 to machine
  precision.
- Explain why a phase shifter makes $Y_\text{bus}$ non-symmetric and how a Ward equivalent is a Schur
  complement of $Y_\text{bus}$.

## Resources

- Free: Overbye, TAMU ECEN 615 lecture slides <https://overbye.engr.tamu.edu/course-2/>; NPTEL *Power System
  Analysis* (A. K. Sinha, IIT Kharagpur) <https://nptel.ac.in/courses/108105067>; MATPOWER User's Manual
  (branch model) <https://matpower.org/docs/MATPOWER-manual.pdf>; pandapower element documentation
  <https://pandapower.readthedocs.io/>.
- Software: MATPOWER <https://matpower.org/> (source <https://github.com/MATPOWER/matpower>), which runs in GNU
  Octave <https://octave.org/>. The MATPOWER 8.1 manual lists GNU Octave 6.2 or later for all features
  (§2.1); §3.2 describes the branch model and §3.6 the network equations used below.
- Textbooks:
  - Grainger & Stevenson, *Power System Analysis* (12 syllabi, tied as the most cited). 1994 **BORROW**
    [`powersystemanaly0000grai_w3w3`](https://archive.org/details/powersystemanaly0000grai_w3w3).
    ES: *Análisis de sistemas de potencia* (McGraw-Hill, 1996).
  - Glover, Sarma, Overbye & Birchfield, *Power System Analysis and Design* (12 syllabi; 7th ed. 2023
    current). Recent editions **PRINT-DISABLED**; 1st ed. 1987 **BORROW**
    [`powersystemanaly0000glov`](https://archive.org/details/powersystemanaly0000glov).
    ES: *Sistemas de potencia: análisis y diseño* (Thomson, 3rd ed. 2004).
  - Gómez-Expósito, Conejo & Cañizares (eds.), *Electric Energy Systems: Analysis and Operation* (8 syllabi;
    the European choice). **NOT ON IA**. ES (original): *Análisis y operación de sistemas de energía
    eléctrica* (McGraw-Hill, 2002).
  - Saadat, *Power System Analysis* (4 syllabi; MATLAB-based). **BORROW**
    [`powersystemanaly0000saad`](https://archive.org/details/powersystemanaly0000saad).
  - Kothari & Nagrath, *Modern Power System Analysis* 4th ed. **BORROW**
    [`modernpowersyste0000koth`](https://archive.org/details/modernpowersyste0000koth).
  - Stagg & El-Abiad, *Computer Methods in Power System Analysis* (1968; 3 syllabi), the classic on network
    matrices and load-flow algorithms. **BORROW**
    [`computermethodsi0000glen`](https://archive.org/details/computermethodsi0000glen).

## Worked example

**Branch model.** MATPOWER (and pandapower, which ports it) models every line, transformer and phase shifter
as a $\pi$ section with series admittance $y_s = 1/(r_s + j x_s)$ and total charging susceptance $b_c$, in
series with an ideal transformer of complex ratio $N = \tau e^{j\theta_\text{shift}}$ at the from end:

$$
\begin{bmatrix} i_f \\ i_t \end{bmatrix} =
\underbrace{\begin{bmatrix}
\left(y_s + j\frac{b_c}{2}\right)\frac{1}{\tau^2} & -\,y_s \frac{1}{\tau e^{-j\theta_\text{shift}}} \\
-\,y_s \frac{1}{\tau e^{j\theta_\text{shift}}} & y_s + j\frac{b_c}{2}
\end{bmatrix}}_{Y_\text{br} = \begin{bmatrix} y_{ff} & y_{ft} \\ y_{tf} & y_{tt} \end{bmatrix}}
\begin{bmatrix} v_f \\ v_t \end{bmatrix}.
$$

With $y_{ft} \neq y_{tf}$ whenever $\theta_\text{shift} \neq 0$, a phase shifter breaks the symmetry of
$Y_\text{bus}$.

**Assembly.** Stack the four entries of all $n_l$ branches into vectors $Y_{ff}, Y_{ft}, Y_{tf}, Y_{tt}$ and let
$C_f, C_t \in \{0,1\}^{n_l \times n_b}$ mark the from and to bus of each branch. With $[\cdot]$ the diagonal
matrix of a vector,

$$
Y_f = [Y_{ff}]\,C_f + [Y_{ft}]\,C_t, \qquad
Y_t = [Y_{tf}]\,C_f + [Y_{tt}]\,C_t, \qquad
Y_\text{bus} = C_f^\top Y_f + C_t^\top Y_t + [Y_\text{sh}].
$$

For plain lines ($\tau = 1$, $\theta_\text{shift} = 0$) this reduces to a complex weighted Laplacian of
[Chapter 3](../part-1-math/ch03-graph-theory.md) plus diagonal terms,

$$
Y_\text{bus} = A^\top [y_s]\, A + \left[(C_f + C_t)^\top \tfrac{j b_c}{2}\right] + [Y_\text{sh}],
\qquad A = C_f - C_t .
$$

Dropping resistances, charging and shunts leaves $Y_\text{bus} = -j\,A^\top \operatorname{diag}(1/x)\, A$, whose
imaginary part (up to sign) is the DC matrix $B_\text{bus} = A^\top \operatorname{diag}(b)\, A$, $b = 1/x$, of
[Chapter 16](ch16-dc-power-flow.md).

**Octave + MATPOWER.** Rebuild $Y_\text{bus}$ for IEEE 14 and compare with `makeYbus`:

```octave
define_constants;
mpc = ext2int(loadcase('case14'));
[Ybus, Yf, Yt] = makeYbus(mpc);

nb = size(mpc.bus, 1);  nl = size(mpc.branch, 1);
Cf = sparse(1:nl, mpc.branch(:, F_BUS), 1, nl, nb);
Ct = sparse(1:nl, mpc.branch(:, T_BUS), 1, nl, nb);
stat = mpc.branch(:, BR_STATUS);
ys = stat ./ (mpc.branch(:, BR_R) + 1j * mpc.branch(:, BR_X));
bc = stat .* mpc.branch(:, BR_B);
tap = ones(nl, 1);  k = find(mpc.branch(:, TAP));  tap(k) = mpc.branch(k, TAP);
tap = tap .* exp(1j * pi / 180 * mpc.branch(:, SHIFT));
Ytt = ys + 1j * bc / 2;  Yff = Ytt ./ (tap .* conj(tap));
Yft = -ys ./ conj(tap);  Ytf = -ys ./ tap;
Ysh = (mpc.bus(:, GS) + 1j * mpc.bus(:, BS)) / mpc.baseMVA;
D = @(v) spdiags(v, 0, numel(v), numel(v));
Y2 = Cf.' * (D(Yff) * Cf + D(Yft) * Ct) + Ct.' * (D(Ytf) * Cf + D(Ytt) * Ct) + D(Ysh);
disp(full(max(max(abs(Ybus - Y2)))))   % expect round-off only
```

The same construction in NumPy against `pandapower.pypower.makeYbus.makeYbus(baseMVA, bus, branch)` on the
`ppc` of `pandapower.networks.case14()` reproduces pandapower's matrix exactly; note that pandapower stores
`TAP = 1` for lines, while MATPOWER case files use `TAP = 0`.

## Lab

!!! example "Lab: IEEE 14 admittance matrix, two ways"

    Tools: Jupyter + NumPy + pandapower, or Octave + MATPOWER (either path is fine; doing both is better).

    1. Load IEEE 14 (`pandapower.networks.case14()` and run `pandapower.runpp` once to populate `net._ppc`, or
       `loadcase('case14')` in Octave). Build $C_f$, $C_t$, $Y_f$, $Y_t$ and $Y_\text{bus}$ from the branch and
       bus tables as in the worked example.
    2. Compare with `pandapower.pypower.makeYbus.makeYbus` (or MATPOWER's `makeYbus`) and report the largest
       absolute difference.
    3. Plot the sparsity pattern of $Y_\text{bus}$ with `matplotlib.pyplot.spy` (Octave: `spy`). Identify the
       transformer branches from their off-nominal taps and check the symmetry of $Y_\text{bus}$ numerically.
    4. Set a 5° phase shift on one transformer, rebuild, and measure $\lVert Y_\text{bus} - Y_\text{bus}^\top \rVert$.

## Exercises

1. Build $Y_\text{bus}$ by inspection for a three-bus ring with series impedances $j0.1$, $j0.2$, $j0.25$ p.u.
   and charging $b_c = 0.02$ p.u. on each line. Check that every row sums to the bus's total shunt admittance.
2. A transformer with ratio $\tau = 1.05$ and no phase shift connects buses 1 and 2 with $x = 0.1$ p.u. Write its
   $Y_\text{br}$ and draw the equivalent $\pi$ circuit implied by it. Why is the $\pi$ circuit asymmetric?
3. *(Octave + MATPOWER)* For `case14`, compute $B_\text{bus} = A^\top \operatorname{diag}(1/x)\, A$ ignoring taps and
   compare it with the first output of `makeBdc(mpc)`. Explain every entry that differs.
4. *(Jupyter: NumPy)* Kron reduction: partition the buses of IEEE 14 into kept and eliminated sets, compute the
   Ward-type equivalent $Y_{kk} - Y_{ke} Y_{ee}^{-1} Y_{ek}$, and show that it reproduces the kept-bus voltages
   for a current injection pattern with zero injection at the eliminated buses.
5. *(Jupyter: NumPy + matplotlib)* For a 300 km line with given per-km parameters, compare the exact long-line
   ABCD parameters with the nominal $\pi$ model as a function of length, and plot the relative error of the
   series admittance.
