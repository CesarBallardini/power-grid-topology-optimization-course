# Chapter 11 – Solutions

Solutions of the Lab and the Exercises of [Chapter 11](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`dc_nodal.py`](src/dc_nodal.py) (the `nodal_solve` of the Chapter 8 lab,
with susceptances), [`pst_flows.py`](src/pst_flows.py) (the worked example, lab steps 3 to 5),
[`pst_phasors.py`](src/pst_phasors.py) (lab steps 1 and 2, Exercise 5), [`trafo_equivalents.py`](src/trafo_equivalents.py)
and [`trafo_equivalents.m`](src/trafo_equivalents.m) (Exercises 1 to 3), [`tap_tables.py`](src/tap_tables.py)
(Exercise 4) and [`hvdc_ring.py`](src/hvdc_ring.py) (Exercise 6). Each Python module prints every number quoted
below when run as a script; the worked examples run live in [examples.ipynb](src/examples.ipynb). The GeoGebra
steps were not run here: for them the solution gives the construction and the values it must show, checked
numerically.

## Lab

1. **GeoGebra: quadrature booster against symmetric PST.** `W_q = U + k i U` adds a voltage perpendicular to
   `U`, so $W_q = U(1 + jk)$: its argument is $\alpha = \arctan k$ and its magnitude $|U|\sqrt{1 + k^2}$, always
   larger than $|U|$. `W_s = U (cos(α) + i sin(α))` is $U e^{j\alpha}$, a pure rotation: `abs(W_s)` stays at 1 for
   every `k`. `pst_phasors.quadrature_booster` and `symmetric_pst` give, for $k = 0.05$, $0.1$ and $0.2$, shifts of
   $2.862^\circ$, $5.711^\circ$ and $11.310^\circ$ with $|W_q| = 1.0012$, $1.0050$ and $1.0198$, and $|W_s| = 1.0000$
   in each case. The symmetric design keeps the magnitude because its series voltage is not perpendicular to
   the input but to the *bisector* of input and output: the triangle $U$, $W_s$, $W_s - U$ is isosceles, so the
   output lies on the same circle as the input. A quadrature booster is cheaper (one series winding fed from a
   quadrature voltage) and its small magnitude change, second order in $k$, is accepted. *(Construction not
   run here.)*
2. **GeoGebra: the general series voltage.** `W_g = U + k (cos(φ) + i sin(φ))` with $U = 1$ has argument
   $\operatorname{atan2}(k\sin\varphi,\ 1 + k\cos\varphi)$, the formula of Chapter 1, Exercise 4, and magnitude
   $\sqrt{(1 + k\cos\varphi)^2 + (k\sin\varphi)^2}$. `pst_phasors.series_voltage(1, 0.05, phi)` against
   `phase_shift_deg(0.05, phi)`: at $\varphi = 90^\circ$ both give $2.862^\circ$ with $|W_g| = 1.0012$ (the quadrature
   booster of step 1), at $\varphi = 60^\circ$ both give $2.419^\circ$ with $|W_g| = 1.0259$: a step angle below
   $90^\circ$ buys less shift and more magnitude change for the same winding. The two values $2.86^\circ$ and
   $2.42^\circ$ are those of Chapter 1 ($n = 5$ taps of 1 %). *(Construction not run here.)*
3. **$F_a$ and $F_b$ against $\alpha$.** `pst_flows.two_path_flows` evaluates the closed formula:

    | $\alpha$ | $F_a$ (pu) | $F_b$ (pu) |
    |---|---|---|
    | $-10^\circ$ | 0.3137 | 0.6863 |
    | $-5^\circ$ | 0.5318 | 0.4682 |
    | $0^\circ$ | 0.7500 | 0.2500 |
    | $5^\circ$ | 0.9682 | 0.0318 |
    | $10^\circ$ | 1.1863 | $-0.1863$ |

    Both lines are straight: $F_a$ rises by $0.0436$ pu per degree (`slope_per_degree`, $(\pi/180)/0.4$) and
    $F_b = P - F_a$ falls by the same amount; the sum is always 1 pu. Line $b$ carries nothing when $F_a = P$,
    that is when $\theta_{12} = 0$: then $F_a = \alpha / x_a = P$ gives $\alpha = x_a P = 0.1$ rad $= 5.73^\circ$
    (`alpha_for_zero_flow_b`). Beyond it $F_b$ is negative: the PST pushes more than the whole transfer through
    line $a$ and the surplus comes back through line $b$, a loop flow in the literal sense. At $10^\circ$ line $a$
    carries 119 % of the transfer.
4. **The PST as a pair of injections.** With the PST removed, line $a$ is a plain reactance and the network
   has injections $P - \alpha/x_a$ at bus 1 and $-(P - \alpha/x_a)$ at bus 2; `nodal_solve` gives
   $\theta_{12} = (P - \alpha/x_a) / (1/x_a + 1/x_b)$ and the flows $\theta_{12}/x_a$, $\theta_{12}/x_b$. Adding
   $\alpha/x_a$ back to line $a$ gives $(\theta_{12} + \alpha)/x_a$, the PST branch's true flow:

    ```python title="pst_flows.py (injections and PSDF)"
    --8<-- "docs/part-2-circuits/ch11-transformers-psts-hvdc/src/pst_flows.py:injections"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch11-transformers-psts-hvdc/src){ target="_blank" } to see or download all the files of this chapter.

    `two_path_by_injections` agrees with `two_path_flows` to $10^{-12}$ at every angle tested (the test runs
    $-10^\circ$, $-3^\circ$, $0^\circ$, $4.5^\circ$ and $10^\circ$). The equivalence is exact in the DC model because
    the PST equation $F_a = (\theta_{12} + \alpha)/x_a$ is linear: the term $\alpha/x_a$ is a constant flow
    entering bus 2 and leaving bus 1 through branch $a$, and a constant flow through a branch is
    indistinguishable, for the rest of the network, from an injection pair at its ends. This is why ToOp can keep
    one nodal matrix for all taps and move the PST angle into the injection vector (`dc_solver/jax/pst.py`).
5. **A PSDF column.** `psdf_column(FR, TO, B, 3, INJ, 6)` puts the PST on branch `(1, 3)` of the Chapter 2
   network (branch index 3) and takes the difference quotient of the flows for a step of one degree:

    | Branch | base flow (pu) | change per degree (pu) |
    |---|---|---|
    | (0, 1) | 0.3041 | $+0.00612$ |
    | (0, 2) | 0.1959 | $-0.00612$ |
    | (1, 2) | 0.0702 | $-0.01470$ |
    | (1, 3) | 0.2339 | $+0.02082$ |
    | (2, 4) | 0.7661 | $-0.02082$ |
    | (3, 4) | 0.2339 | $+0.02082$ |
    | (4, 5) | 1.0000 | $0$ |

    The column is a circulation: it adds up to zero at every node (`test_psdf_column_is_exact_and_step_independent`
    checks $A^\top \Delta F = 0$), so it changes no injection, only the split of the flows among parallel paths.
    The PST pushes $0.0208$ pu per degree through its own branch and on through (3, 4); the same amount returns
    through (2, 4), then back to node 1 partly through (1, 2) ($0.0147$) and partly around node 0 ($0.0061$ through
    (0, 2) and (0, 1)). Branch (4, 5) does not react: node 5 is a leaf, so KCL fixes its flow at the 1 pu
    withdrawn there, whatever any PST does. The column is the same for a step of $0.01^\circ$ and the same for
    zero injections (the test checks both), which is the point: in the linear DC model the sensitivity does not
    depend on the operating point, so ToOp computes it once in preprocessing (`dc_solver/preprocess/helpers/psdf.py`)
    and scales it by `base_mva * (np.pi / 180)` to get MW per degree, $2.08$ MW per degree on 100 MVA for
    branch (1, 3) here.

## Exercise 1

An impedance $Z$ on the low-voltage side draws $I_2 = V_2 / Z$; the ideal transformer carries $V_1 = (380/110) V_2$
and $I_1 = (110/380) I_2$, so from the high-voltage side

$$
Z' = \frac{V_1}{I_1} = \left(\frac{380}{110}\right)^2 \frac{V_2}{I_2} = \left(\frac{380}{110}\right)^2 Z = 11.934 \times 10 = 119.3\ \Omega
$$

(`trafo_equivalents.referred_impedance(10, 380, 110)` gives $119.34\ \Omega$). Impedances scale with the square of
the ratio because voltage goes up by the ratio and current down by it. This is also why per-unit values do not
depend on the side (Chapter 10, Exercise 4): the base impedance scales by the same square.

## Exercise 2

The DC susceptance of a branch is $b = 1/x$. Replacing $x$ by $x/\rho$ gives $b' = \rho / x = \rho\, b$, a change
of $\rho - 1 = +5$ % for $\rho = 1.05$ (`susceptance_change`). A larger susceptance means a stiffer branch, so in a
meshed network the transformer attracts *more* flow: in the two-path example with the transformer ($x = 0.1$)
in parallel with a line ($x = 0.3$), its share of a 1 pu transfer rises from $0.7500$ to $0.7590$
(`flow_share_with_ratio`), a change of $+1.2$ %, less than 5 % because the parallel path dilutes it:
$\rho b_t / (\rho b_t + b_\ell)$ against $b_t / (b_t + b_\ell)$. A $\rho$ below 1 would do the opposite. The sign of the
effect depends on which side the reactance is referred to: PowSyBl's `dc_use_transformer_ratio = True` and
ToOp's `get_trafos` make the same choice, which is what the comment in the code records.

## Exercise 3

$$
x_1 = \frac{0.10 + 0.25 - 0.20}{2} = 0.075, \qquad
x_2 = \frac{0.10 + 0.20 - 0.25}{2} = 0.025, \qquad
x_3 = \frac{0.25 + 0.20 - 0.10}{2} = 0.175\ \text{pu},
$$

and indeed $x_1 + x_2 = x_{12}$, $x_1 + x_3 = x_{13}$, $x_2 + x_3 = x_{23}$: each pairwise short-circuit test sees the
two star branches in series, which is where the formula comes from (`trafo_equivalents.star_equivalent`; the
Octave version is `trafo_equivalents.m`). A star branch can come out negative in real units when one winding
sits between the other two; it is still a valid branch of the equivalent, only not a physical reactance.

The star node is not a busbar: it is a fictitious point inside one tank. A fault or a maintenance outage of the
transformer removes the whole unit, so the three branches that meet at the star node disappear together, and
leaving the star node connected to one of them would create a dangling node with no injection. That is why
the pandapower backend turns the outage of a three-winding transformer into a multi-outage of three branches
(`get_trafo3w_multioutage`), and why the PowSyBl backend simply refuses networks that contain one.

## Exercise 4

`tap_tables.constant_x_table()` builds taps $-16 \dots 16$ with $x = 0.1$ pu throughout and
$\alpha(n) = \operatorname{atan2}(0.01 n \sin 90^\circ,\ 1 + 0.01 n \cos 90^\circ) = \arctan(0.01 n)$;
`growing_x_table()` keeps the same $\alpha$ and sets $x = 0.1\,(1 + 0.01 |n|)$, $0.116$ pu at the end taps.
`is_linear_pst` applies `np.allclose(x, x[0])`: `True` for the first table, `False` for the second. ToOp treats
the first as linear in DC (one PSDF column, the angle in the injection vector) and needs per-tap susceptances
for the second, as the chapter's ToOp note says.

$\alpha$ is *not* exactly proportional to the tap number in the first table: $\arctan$ grows more slowly than its
argument. The shift per tap is $0.5729^\circ$ at tap 1 and only $0.5681^\circ$ at tap 16, where
$\alpha = 9.090^\circ$ against $16 \times 0.5729^\circ = 9.167^\circ$ if it were proportional, a shortfall of
$0.08^\circ$ (0.8 %). It is proportional to third order, $\alpha \approx 0.01 n - (0.01 n)^3/3$, which is why the
first few taps look linear and the end taps do not. Since ToOp reads the angle of the selected tap from a table
instead of multiplying a step by a tap number, this non-linearity costs it nothing; "linear" in `get_linear_pst`
refers to the reactance, not to the angle.

## Exercise 5

*(GeoGebra construction not run here.)* Draw $U = 1$, then for the quadrature booster the series voltage
$k\, jU$ perpendicular to $U$ and the output $W_q = U + jkU$; for the symmetric PST the output $W_s = e^{j\alpha} U$
on the unit circle. The two outputs have the same argument when $k = \tan\alpha$, so $|W_q| = \sqrt{1 + \tan^2\alpha} = 1/\cos\alpha$
and $|W_s| = 1$ (`pst_phasors.quadrature_magnitude_for_shift`): for $\alpha = 5^\circ$, $10^\circ$ and $20^\circ$
the quadrature output measures $1.0038$, $1.0154$ and $1.0642$ pu, the symmetric one $1.0000$. In the diagram
the symmetric PST's series voltage $W_s - U$ is the chord of the unit circle, perpendicular to the bisector of
the angle $\alpha$ (so it is at $90^\circ + \alpha/2$ from $U$, not at $90^\circ$), and its length $2\sin(\alpha/2)$ is
slightly shorter than the quadrature booster's $\tan\alpha$.

## Exercise 6

`hvdc_ring.py` uses the ring $x_{12} = 0.10$, $x_{23} = 0.15$, $x_{13} = 0.20$ pu with a generator at bus 1
feeding loads of 0.6 pu at bus 2 and 0.4 pu at bus 3. The HVDC link is `inj[0] -= p_dc`, `inj[2] += p_dc` before
`nodal_solve` (bus 1 is the slack, so the withdrawal at bus 1 is absorbed by the balance anyway). At $P_{dc} = 0$
the AC flows are $0.6444$ pu on L1-2, $0.0444$ on L2-3 and $0.3556$ on L1-3, and L1-2 is the most loaded line.
Because the DC model is linear, $F(P_{dc}) = F(0) + P_{dc}\, d$ with $d = F(1) - F(0) = (-0.4444, -0.4444, -0.5556)$:
every unit the link carries is a unit that no longer travels over the AC ring, $0.444$ of it along the path
1–2–3 and $0.556$ along L1-3, in inverse proportion to the path reactances ($0.25$ and $0.20$). The test checks
the affine law at three setpoints and the script at nine: the largest deviation is $2 \times 10^{-16}$.

Unloading the most loaded line can mean two things. The flow on L1-2 reaches zero at $P_{dc} = 0.6444/0.4444 = 1.45$
pu, but by then L1-3 carries $-0.45$ pu backwards, so the line that was relieved has handed the problem to its
neighbour. The sensible target is the setpoint that minimizes the *largest* AC flow, a one-variable linear
program (`best_setpoint`, solved with `scipy.optimize.linprog`): $P_{dc} = 0.775$ pu, where L1-2 carries
$+0.300$ pu, L2-3 $-0.300$ pu and L1-3 $-0.075$ pu. The maximum AC loading falls from $0.644$ to $0.300$ pu, and
moving the setpoint $0.1$ pu either way makes it worse again (the test checks both sides). The figure in
`examples.ipynb` shows the three straight lines crossing and the setpoint where $|F_{12}| = |F_{23}|$.

The HVDC setpoint is a continuous action: $P_{dc}$ can take any value between the converter limits and every
flow responds linearly and smoothly, so the optimization is a small linear program with a unique, interior
optimum, as above. A busbar split is a discrete choice among configurations (Chapter 19 counted them):
there is no "half a split", the flows jump from one topology to the next, and the problem becomes combinatorial.
That difference is why ToOp's action-set schema reserves `HVDCRange` with `min_power` and `max_power` for the
continuous variable, even though the solver does not optimize it yet, while busbar splits and PST taps are
enumerated.
