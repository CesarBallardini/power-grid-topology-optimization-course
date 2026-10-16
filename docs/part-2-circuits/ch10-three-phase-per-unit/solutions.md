# Chapter 10 – Solutions

Solutions of the Lab and the Exercises of [Chapter 10](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`per_unit.py`](src/per_unit.py) (bases, change of base, the branch table
and the MW limit of the lab, Exercise 4), [`three_phase.py`](src/three_phase.py) (balanced phasors, the wye load,
instantaneous power, the relay impedance and the symmetrical components of Exercises 1, 2, 3 and 5) and
[`sym_components.m`](src/sym_components.m) (Exercise 5 in Octave, as the statement asks). Each Python module
prints every number quoted below when run as a script; the worked examples run live in
[examples.ipynb](src/examples.ipynb). The GeoGebra steps were not run here: for them the solution gives the
construction and the values it must show, checked numerically.

## Lab

1. **GeoGebra: line and phase voltages.** With $V_a = 1$, $V_b = e^{-j120^\circ}$ and $V_c = e^{j120^\circ}$, the
   difference $V_{ab} = V_a - V_b = 1 - (-\tfrac12 - j\tfrac{\sqrt3}{2}) = \tfrac32 + j\tfrac{\sqrt3}{2}$ has
   magnitude $\sqrt{9/4 + 3/4} = \sqrt3$ and argument $\arctan(\tfrac{\sqrt3}{2} / \tfrac32) = 30^\circ$:
   `abs(V_ab)` shows 1.7321 and `arg(V_ab)` shows $30^\circ$ (`three_phase.line_voltages(balanced_set())`). The
   geometry is the isosceles triangle $V_a$, $V_b$, $V_{ab}$ with a $120^\circ$ apex: the base is $2 \sin 60^\circ = \sqrt3$
   times the side, and the base is tilted $30^\circ$ from $V_a$, halfway between $V_a$ and $-V_b$. The same holds for
   $V_{bc}$ and $V_{ca}$, each $120^\circ$ further on, so line voltages are also a balanced set, $\sqrt3$ larger and
   $30^\circ$ ahead of the phase voltages. *(Construction not run here.)*
2. **GeoGebra: balanced currents.** `I_a = V_a (cos(-φ) + i sin(-φ))` and its two companions form a balanced set
   rotated by $-\varphi$, so $I_a + I_b + I_c = (1 + a^2 + a)\, e^{-j\varphi} = 0$ for every `φ`: the neutral carries
   nothing (numerically $4 \times 10^{-16}$ for $\varphi = 30^\circ$). The angle between each voltage and its
   current is the same $\varphi$ in all three phases (`balanced_set()[1] / balanced_set(1, -phi)[1]` is at
   $\varphi$), which is why one power factor describes the whole load and why the per-phase equivalent circuit
   of a balanced system loses nothing. *(Construction not run here.)*
3. **The per-unit branch table.** `per_unit.branch_table()` adds the columns to the branch list:

    ```python title="per_unit.py (branch table)"
    --8<-- "docs/part-2-circuits/ch10-three-phase-per-unit/src/per_unit.py:table"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch10-three-phase-per-unit/src){ target="_blank" } to see or download all the files of this chapter.

    | Branch | $Z_{\text{base}}$ ($\Omega$) | $r_{\text{pu}}$ | $x_{\text{pu}}$ | limit (MW) |
    |---|---|---|---|---|
    | 380 kV line, 150 km, 2 000 A | 1 444 | 0.0031 | 0.0312 | 1 316.4 |
    | 220 kV line, 100 km, 1 000 A | 484 | 0.0124 | 0.0826 | 381.1 |
    | 110 kV cable, 20 km, 600 A | 121 | 0.0083 | 0.0198 | 114.3 |

    Two things the table teaches. First, the 380 kV line is 45 $\Omega$ of reactance and the cable 2.4 $\Omega$,
    yet in per unit the cable is "closer" to the line than the ohms suggest, because the base impedance falls
    with the square of the voltage: per unit measures a branch against what its own voltage level can carry.
    Second, the 220 kV line reproduces the worked example ($x_{\text{pu}} = 0.0826$).
4. **ToOp's formula.** `value * limit_voltage * 1e-3 * math.sqrt(3)` is $\sqrt3\, V_{LL} I$ with `value` in
   ampere and `limit_voltage` in kilovolt: A × kV = kVA, and `1e-3` turns kVA into MVA, so the result is the
   three-phase apparent power in MVA at the current limit, used as a MW limit at unity power factor (the
   `dc_power_factor` of Chapter 9). The `p_limit_mw` column uses the same product in the same units and agrees to
   machine precision (`test_toop_formula_matches_table`). The factor $\sqrt3$ is there because the limit is a
   line current and the voltage a line-to-line value; with phase voltages it would be $3 V_{\text{ph}} I$.

## Exercise 1

The phase voltage is $V_{\text{ph}} = 400 / \sqrt3 = 230.94$ V and $|Z_Y| = \sqrt{30^2 + 40^2} = 50\ \Omega$ at
$53.13^\circ$, so the line current (equal to the phase current in a wye) is
$I_L = 230.94 / 50 = 4.619$ A, lagging its phase voltage by $53.13^\circ$. The total power is three times the
per-phase power:

$$
S = 3\, |I_L|^2 Z_Y = 3 \times 21.33 \times (30 + j40) = 1920 + j2560\ \text{VA},
\qquad |S| = 3200\ \text{VA} = \sqrt3 \times 400 \times 4.619 .
$$

$P = 1920$ W, $Q = 2560$ var, power factor $0.6$ lagging. The equivalent delta impedance is $Z_\Delta = 3 Z_Y = 90 + j120\ \Omega$:
a delta branch sees the line voltage, $\sqrt3$ times larger, and carries the phase current, $1/\sqrt3$ of the
line current, so it must be 3 times the wye impedance to draw the same power (`three_phase.wye_load`).

## Exercise 2

For a balanced load, $p_a(t) = VI\cos\varphi + VI\cos(2\omega t - \varphi)$ as in Chapter 9, Exercise 3, and $p_b$,
$p_c$ are the same expression with $\omega t$ shifted by $\mp 120^\circ$, so their pulsating terms are shifted by
$\mp 240^\circ$, that is by $\pm 120^\circ$: three equal cosines $120^\circ$ apart, which add to zero, as the
phasors of lab step 2 do. What remains is $p_a + p_b + p_c = 3 VI \cos\varphi$, constant.
`three_phase.instantaneous_power(230, 10, 30)` samples one period: each phase power swings by $\pm 2300$ W
around its mean while the total stays at $5975.6$ W to nine digits (min and max are both 5 975.6 W,
against $3 \times 230 \times 10 \times \cos 30^\circ = 5975.6$ W). This is the mechanical reason for three-phase
transmission: a three-phase generator delivers constant torque, a single-phase one pulsates at 100 Hz.
The figure in `examples.ipynb` shows the three pulsating curves and the flat total (not saved).

## Exercise 3

$V_{\text{ph}} = 380 / \sqrt3 = 219.39$ kV, so $|Z| = V_{\text{ph}} / I = 219\,393 / 1000 = 219.4\ \Omega$. The angle
is $\operatorname{atan2}(200, 600) = 18.43^\circ$, hence

$$
R = 219.4 \cos 18.43^\circ = 208.1\ \Omega, \qquad X = 219.4 \sin 18.43^\circ = 69.4\ \Omega
$$

(`three_phase.relay_impedance(380, 1000, 600, 200)`). Two remarks on the model. The impedance angle is the
angle of $S = P + jQ$, which is the angle of $V I^*$ and therefore of $V/I$: $\operatorname{atan2}(Q, P)$ is
exactly the impedance angle, and it is positive (inductive) for a lagging load. And the magnitude uses the phase
voltage because a distance relay measures phase voltage over phase current; using $V_{LL}$ would overstate $|Z|$
by $\sqrt3$, the same factor as in lab step 4.

## Exercise 4

Change of base at the same voltage: $x = 0.15 \times 100 / 1600 = 0.009375$ pu, under one percent. The
transformer looks small on the system base because 100 MVA is a small load for a 1 600 MVA unit; the ohms did
not change.

Why the side does not matter: let the ratio be $380/110$ kV with bases $V_{\text{HV}} = 380$ kV and
$V_{\text{LV}} = 110$ kV, chosen through the turns ratio, and the same $S_{\text{base}}$ on both sides. An ohmic
reactance $X_{\text{HV}}$ on the high-voltage side is $X_{\text{LV}} = X_{\text{HV}} (110/380)^2$ referred to the
low-voltage side (Chapter 11, Exercise 1), while the base impedances are in the ratio
$Z_{\text{base,HV}} / Z_{\text{base,LV}} = (380/110)^2$: 90.25 $\Omega$ against 7.5625 $\Omega$ on 1 600 MVA, a
ratio of 11.934 = $(380/110)^2$ (`per_unit.main`). Dividing, $X_{\text{HV}} / Z_{\text{base,HV}} = X_{\text{LV}} / Z_{\text{base,LV}}$:
the referral factor and the base factor are the same $(V_{\text{HV}}/V_{\text{LV}})^2$ and cancel. In per unit the
ideal transformer disappears from the circuit; `test_exercise_4_change_of_base_is_independent_of_side` checks
the two routes to 0.009375. The argument fails as soon as the voltage bases are not in the turns ratio (an
off-nominal tap, say): then a residual ratio stays in the model, which is what ToOp's $x/\rho$ in Chapter 11
accounts for.

## Exercise 5

With $a = e^{j120^\circ}$ and $V_{abc} = (1,\ 0.9 a^2,\ a)$, $A^{-1} = \tfrac13 \begin{bmatrix} 1 & 1 & 1 \\ 1 & a & a^2 \\ 1 & a^2 & a \end{bmatrix}$
gives

$$
V_0 = \tfrac13 (1 + 0.9 a^2 + a) = -\tfrac{0.1}{3} a^2, \quad
V_1 = \tfrac13 (1 + 0.9 a^3 + a^3) = \tfrac{2.9}{3}, \quad
V_2 = \tfrac13 (1 + 0.9 a^4 + a^2) = -\tfrac{0.1}{3} a,
$$

using $a^3 = 1$ and $1 + a + a^2 = 0$: $|V_1| = 0.9667$ and $|V_0| = |V_2| = 0.0333$ pu. The 10 % dip of one phase is
mostly a 3.3 % loss of positive sequence plus equal 3.3 % zero and negative sequences, which is how protection
sees an unbalance. `sym_components.m` is the Octave solution the statement asks for; `three_phase.symmetrical_components`
gives the same numbers in Python and `examples.ipynb` runs it.

```octave title="sym_components.m"
--8<-- "docs/part-2-circuits/ch10-three-phase-per-unit/src/sym_components.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch10-three-phase-per-unit/src){ target="_blank" } to see or download all the files of this chapter.
