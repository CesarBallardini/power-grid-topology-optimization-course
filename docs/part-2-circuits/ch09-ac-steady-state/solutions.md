# Chapter 9 – Solutions

Solutions of the Lab and the Exercises of [Chapter 9](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`ac_transfer.py`](src/ac_transfer.py) (power through a reactance, the
error table, Exercises 4 and 6), [`rl_load.py`](src/rl_load.py) and [`rl_load.m`](src/rl_load.m) (Exercise 1),
[`instantaneous_power.py`](src/instantaneous_power.py) (Exercise 3), [`ac_nodal.py`](src/ac_nodal.py) and
[`ac_nodal.m`](src/ac_nodal.m) (Exercise 5). Each Python module prints every number quoted below when run as a
script; the worked examples run live in [examples.ipynb](src/examples.ipynb). The GeoGebra steps were not run
here: for them the solution gives the construction and the values it must show, checked numerically.

## Lab

1. **GeoGebra: the P–δ curve.** `p(x) = V_1 V_2 sin(x) / X_L` is a sine of amplitude $V_1 V_2 / X_L$: the
   maximum transferable power is $V_1 V_2 / X_L$ and it occurs at $x = 90^\circ$, whatever the sliders say. With
   $V_1 = V_2 = 1$ and $X_L = 0.1$ the peak is $10$ pu (`transfer_power(1, 1, 0.1, 90.0)`); raising either voltage
   raises the peak in proportion, raising $X_L$ lowers it, and none of them moves the peak away from $90^\circ$.
   The line `q(x) = x / X_L` is the tangent of `p` at the origin: it has no maximum, which is the first thing
   the DC model loses. Beyond $90^\circ$ the exact curve falls while the tangent keeps growing, so the DC model
   can predict transfers that no AC steady state can carry. *(Construction not run here.)*
2. **GeoGebra: the phasor diagram.** Dividing a complex number by $jX_L$ multiplies its magnitude by $1/X_L$ and
   rotates it by $-90^\circ$, so `I_L` is always perpendicular to `U_1 - U_2`, lagging it by a quarter turn:
   `phasor_current(1, 1, 0.1, 30.0)` is at $-90.0^\circ$ from $U_1 - U_2$. As `d` grows from 0, $U_1 - U_2$
   turns from pointing "upwards" (at $90^\circ + d/2$, since the chord of two equal phasors is perpendicular to
   their bisector) and $I_L$ therefore sits at $d/2$: the current is in phase with the bisector of the two
   voltages, halfway between them. Its angle relative to $U_1$ is $-d/2$: the current lags the sending voltage by
   half the angle difference, so the sending end delivers both active power ($I_L$ has a component along $U_1$)
   and reactive power ($I_L$ has a component $90^\circ$ behind $U_1$), as the worked example computes.
   *(Construction not run here.)*
3. **The error table.** `error_table()` tabulates $\delta$, the exact $P = \sin\delta / 0.1$, the DC estimate
   $\delta_{\text{rad}} / 0.1$ and the relative error $\delta / \sin\delta - 1$, which does not depend on $X$:

    ```python title="ac_transfer.py (error_table)"
    --8<-- "docs/part-2-circuits/ch09-ac-steady-state/src/ac_transfer.py:table"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch09-ac-steady-state/src){ target="_blank" } to see or download all the files of this chapter.

    | $\delta$ | exact $P$ (pu) | DC $\delta/X$ (pu) | error |
    |---|---|---|---|
    | $5^\circ$ | 0.872 | 0.873 | 0.13 % |
    | $10^\circ$ | 1.736 | 1.745 | 0.51 % |
    | $20^\circ$ | 3.420 | 3.491 | 2.06 % |
    | $30^\circ$ | 5.000 | 5.236 | 4.72 % |
    | $45^\circ$ | 7.071 | 7.854 | 11.07 % |
    | $60^\circ$ | 8.660 | 10.472 | 20.92 % |
    | $90^\circ$ | 10.000 | 15.708 | 57.08 % |

    The check values hold: 0.51 % at $10^\circ$, 4.72 % at $30^\circ$, 20.92 % at $60^\circ$. The error is
    always positive (the DC model overestimates the flow, since $\sin\delta < \delta$) and grows like
    $\delta^2 / 6$ for small angles, which is why the plot of the error against $\delta$ is a parabola near the
    origin: doubling the angle quadruples the error. Transmission lines in normal operation run at angle
    differences of a few degrees, where the error is below one percent.
4. **Unequal voltages.** With $V_1 = 1.05$ and $V_2 = 0.95$ pu the exact power is $0.9975 \sin\delta / X$ while
   the DC estimate stays $\delta / X$, so the total error factor splits exactly as a product:

    $$
    1 + \varepsilon_{\text{total}} = \frac{\delta / X}{V_1 V_2 \sin\delta / X}
    = \underbrace{\frac{\delta}{\sin\delta}}_{1 + \varepsilon_{\text{angle}}}
    \cdot \underbrace{\frac{1}{V_1 V_2}}_{1 + \varepsilon_{\text{magnitude}}} .
    $$

    `error_table(1.05, 0.95)` gives, at $10^\circ$, $30^\circ$ and $60^\circ$, total errors of 0.76 %, 4.98 % and
    21.22 %, of which the angle part is 0.51 %, 4.72 % and 20.92 % (the same as in step 3) and the magnitude
    part is a constant $1/0.9975 - 1 = 0.25$ %. The magnitude error does not depend on $\delta$ and is small as
    long as the voltages are within the usual $\pm 5$ % band: it is the product $V_1 V_2$ that matters, and a
    high and a low voltage largely compensate ($1.05 \times 0.95 = 0.9975$). Two low voltages would not:
    $0.95^2 = 0.9025$ gives a 10.8 % magnitude error, the situation where the DC model is least trustworthy.

## Exercise 1

$\omega = 2\pi \cdot 50 = 314.16$ rad/s, so $X_L = \omega L = 15.71\ \Omega$ and $Z = 10 + j15.71\ \Omega$,
$|Z| = 18.62\ \Omega$ at $57.5^\circ$. With $V = 230\angle 0$ V, $I = V / Z = 12.35\angle{-57.5^\circ}$ A. The complex
power is $S = V I^* = |I|^2 Z = 152.6 \times (10 + j15.71) = 1526 + j2396$ VA, $|S| = 2841$ VA, and the power factor
is $P / |S| = 10 / 18.62 = 0.537$ lagging. To reach $\text{pf} = 0.95$ the load must draw only
$Q' = P \tan(\arccos 0.95) = 1526 \times 0.3287 = 502$ var, so a capacitor in parallel must supply
$Q_C = 2396 - 502 = 1895$ var:

$$
C = \frac{Q_C}{\omega V^2} = \frac{1895}{314.16 \times 230^2} = 114.0\ \mu\text{F}.
$$

`rl_load.py` prints these numbers and checks them by recomputing $S$ with the capacitor in parallel: $1526 + j501$
VA, pf $0.950$ (the capacitor leaves $P$ untouched). The Octave version is `rl_load.m`:

```octave title="rl_load.m"
--8<-- "docs/part-2-circuits/ch09-ac-steady-state/src/rl_load.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch09-ac-steady-state/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 2

*(GeoGebra construction not run here.)* With sliders `V_m`, `ω`, `φ` and an animated `t`, the point
`u = V_m (cos(ω t + φ) + i sin(ω t + φ))` is $u(t) = V_m e^{j(\omega t + \varphi)} = V_m e^{j\varphi}\, e^{j\omega t}$:
a fixed complex number $V_m e^{j\varphi}$ multiplied by a unit rotation. At $t = 0$ the rotation is $e^{0} = 1$ and
the point is exactly $V_m e^{j\varphi}$, the phasor: a photograph of the rotating vector when the clock starts.
The traced point `(t, x(u))` draws $V_m \cos(\omega t + \varphi)$, the signal: its value at $t = 0$ is
$V_m \cos\varphi$, the real part of the phasor, and its first positive peak is at $t = -\varphi / \omega$, earlier
for a positive $\varphi$ (a leading signal). Moving `φ` rotates the phasor and slides the trace along $t$ without
changing its shape; moving `ω` changes how fast the vector turns and how compressed the trace is, but not the
phasor. That is why phasor analysis can drop $e^{j\omega t}$ altogether: in a linear circuit at one frequency,
every voltage and current rotates at the same rate, and only the photographs at $t = 0$ differ.

## Exercise 3

$$
p(t) = 2VI \cos\omega t \cos(\omega t - \varphi) = VI\cos\varphi + VI\cos(2\omega t - \varphi),
$$

by the product-to-sum identity: a constant term $P = VI\cos\varphi$ and a cosine at twice the supply frequency
whose amplitude is $VI = S$, regardless of $\varphi$. `instantaneous_power.sample_power(230, 10, 36.87)` samples one
period at 50 Hz with 2 000 points; `mean_and_swing` returns a mean of 1840.0 W (against $VI\cos\varphi = 1840.0$ W)
and a swing $(\max - \min)/2$ of 2300.0 VA (against $VI = 2300$ VA). The minimum of $p(t)$ is $-460$ W: twice per
period the load returns energy to the source, which is what a non-unity power factor means in the time domain.
The test also checks that the spectrum of $p(t) - P$ peaks at the second harmonic. A plot of $p(t)$ with the mean
drawn as a horizontal line shows the 100 Hz ripple around $P$ (figure in `examples.ipynb`, not saved).

## Exercise 4

From the worked example, $Q_{12} = (V_1^2 - V_1 V_2 \cos\delta) / X$. At $\delta = 0$, $Q_{12} = V_1 (V_1 - V_2) / X$,
which is positive when $V_1 > V_2$ and negative otherwise: reactive power leaves the bus whose voltage magnitude
is higher. `transfer_power(1.05, 1.0, 0.1, 0.0)` gives $Q_{12} = +0.525$ pu and `transfer_power(1.0, 1.05, 0.1, 0.0)`
gives $-0.500$ pu (the two are not opposite numbers because $Q$ at the sending end also includes the reactive
power absorbed by the reactance itself, $|I|^2 X$).

The DC approximation makes two assumptions, and each one kills $Q$ in its own way. First, flat voltages
$V_1 = V_2 = 1$: then $Q_{12} = (1 - \cos\delta) / X \approx \delta^2 / (2X)$, second order in the angle, while
$P_{12} \approx \delta / X$ is first order. At $\delta = 10^\circ$ with $X = 0.1$ pu, $Q_{12} = 0.152$ pu against
$P_{12} = 1.74$ pu, and a linearization around $\delta = 0$ keeps only first-order terms, so $Q$ drops to zero.
Second, the DC model does not solve for voltage magnitudes at all: it has no unknown that reactive power could
determine. In the AC equations $Q$ is tied to the magnitudes (as the first part of the exercise shows) and $P$
to the angles; the DC model keeps the $P$–$\theta$ half and discards the $Q$–$V$ half. The price is that the DC
limits ToOp uses must be given in MW, which is Exercise 6.

## Exercise 5

Nodes 0 (ground), 1 and 2. The circuit in `ac_nodal.py` has a voltage source holding node 1 at
$V_1 = 230\angle 0$ V, a current source injecting $I_2 = 5\angle{-30^\circ}$ A into node 2, an RL branch
$Z_{RL} = 2 + j5\ \Omega$ between nodes 1 and 2 and an RC branch $Z_{RC} = 10 - j4\ \Omega$ from node 2 to ground.
The admittance matrix is $Y = A^\top \operatorname{diag}(1/Z)\, A$, exactly as in Chapter 8 with $1/Z$ instead
of $g$. Node 1 has a known voltage, so its row leaves the system and its column goes to the right-hand side:
KCL at node 2 alone reads $Y_{22} V_2 = I_2 - Y_{21} V_1$, giving $V_2 = 225.14\angle{-22.8^\circ}$ V. The product
$Y V$ then returns the net injection at every node; at node 1 it is the current the voltage source must
supply, $I_1 = 16.71\angle 7.4^\circ$ A, which is also the RL branch current, and the RC branch carries
$20.90\angle{-1.0^\circ}$ A.

The power balance is Tellegen's theorem in complex form: $\sum_k V_k I_k^* = \sum_b |I_b|^2 Z_b$. Both sides are
$4927.9 - j351.9$ VA: the sources deliver 4 928 W, all dissipated in the two resistors, and the circuit as a whole
is slightly capacitive (the RC branch's $-j4\ \Omega$ with its larger current outweighs the $+j5\ \Omega$ of the
RL branch). The test checks the balance to $10^{-9}$. The same circuit in Octave, with `Y \ I` replaced by the
one-unknown division, is `ac_nodal.m`:

```octave title="ac_nodal.m"
--8<-- "docs/part-2-circuits/ch09-ac-steady-state/src/ac_nodal.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch09-ac-steady-state/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 6

The branch can carry at most $|S| = V I_{\max}$. At power factor 0.95 the admissible active power is
$P_{\text{adm}} = 0.95\, V I_{\max}$, while the DC model allows $P_{\max} = V I_{\max}$. The overestimate, relative to
what the branch can really carry, is

$$
\frac{P_{\max}}{P_{\text{adm}}} - 1 = \frac{1}{0.95} - 1 = 5.26\ \%
$$

(`dc_limit_overestimate(0.95)`), or 5 % if measured against $P_{\max}$ itself. A DC flow at exactly 100 % of the
limit therefore corresponds to a current 5.3 % above the rating once the reactive part is added, which is one of
the reasons ToOp's DC results are checked with an AC power flow before they are trusted
([Chapter 19](../../part-3-power-systems/ch19-operational-security/index.md)).
