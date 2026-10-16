# Chapter 1 – Solutions

Solutions of the Lab and the Exercises of [Chapter 1](index.md). Try each task before reading its solution. The
code is in the chapter's `src` folder: [`phasor_power.py`](src/phasor_power.py) and
[`phasor_power.m`](src/phasor_power.m) (the worked example), [`complex_plane.py`](src/complex_plane.py) (every
numeric check below, plain `cmath`), [`three_loads.m`](src/three_loads.m) (Lab step 4 in Octave) and
[`tap_phase_shift.m`](src/tap_phase_shift.m) (Exercise 4 in Octave). `python complex_plane.py` prints every
number quoted here; the Octave files carry their own `%!test` blocks and were run with Octave 11.3. The GeoGebra
steps were not run here: they are given as the construction plus the values the screen must show, and the same
values are computed by the code.

## Lab

1. **Rotation by a slider.** With `z = 3 + 4i`, `abs(z) = 5` and `arg(z) = 53.13°` ($\arctan(4/3)$). For
   `w = z (cos θ + i sin θ)`, `abs(w)` stays 5 for every slider value and `arg(w) = 53.13° + θ`, wrapped into
   $(-180°, 180°]$ by GeoGebra: `complex_plane.rotate(3 + 4j, θ)` gives $w = 0.598 + 4.964i$ at $\theta = 30°$
   (argument $83.13°$), $-4 + 3i$ at $90°$ ($143.13°$), $-3 - 4i$ at $180°$ ($-126.87° = 233.13° - 360°$) and
   $4 - 3i$ at $270°$ ($-36.87°$). The point $w$ runs around the circle of radius 5; its modulus never changes
   because $|e^{j\theta}| = 1$.

    ```python title="complex_plane.py"
    --8<-- "docs/part-1-math/ch01-complex-numbers/src/complex_plane.py"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch01-complex-numbers/src){ target="_blank" } to see or download all the files of this chapter.

2. **Rotate and scale.** `u = 1 + 0.5i` has `abs(u) = 1.1180` ($\sqrt{1.25}$) and `arg(u) = 26.57°`. The product
   is $p = (3 + 4i)(1 + 0.5i) = 3 + 1.5i + 4i + 2i^2 = 1 + 5.5i$, with `abs(p) = 5.5902` $= 5 \times 1.1180$ and
   `arg(p) = 79.70°` $= 53.13° + 26.57°$ (`complex_plane.product_check`). In one sentence: writing both factors
   in polar form, $r_1 e^{j\theta_1} \cdot r_2 e^{j\theta_2} = (r_1 r_2)\, e^{j(\theta_1 + \theta_2)}$, so
   multiplying by $u$ *scales* $z$ by $|u|$ and *rotates* it by $\arg u$.
3. **Cube roots of unity.** `Sequence(cos(k 120°) + i sin(k 120°), k, 0, 2)` draws $1$, $-0.5 + 0.866i$ and
   $-0.5 - 0.866i$ (`complex_plane.roots_of_unity(3)`). Their sum is $1 - 0.5 - 0.5 + (0.866 - 0.866)i = 0$; in
   floating point the code prints $2 \times 10^{-16}$, which is zero to double precision. Three arrows of equal
   length $120°$ apart close a triangle, which is what "balanced" means for the three phases of
   [Chapter 10](../../part-2-circuits/ch10-three-phase-per-unit/index.md).
4. **Three loads.** With $V = 230$ V and `complex_plane.LOADS` (or `three_loads(230, [40+30i, 40-30i, 50])` in
   Octave):

    | Load | $Z$ ($\Omega$) | $I$ (A) | $S = V I^*$ (VA) | $\lvert S \rvert$ | $\sqrt{P^2 + Q^2}$ | Sign of $Q$ |
    |---|---|---|---|---|---|---|
    | inductive | $40 + j30$ | $3.68 - j2.76$ | $846.4 + j634.8$ | 1058.0 | 1058.0 | $Q > 0$: the current lags, the load absorbs reactive power |
    | capacitive | $40 - j30$ | $3.68 + j2.76$ | $846.4 - j634.8$ | 1058.0 | 1058.0 | $Q < 0$: the current leads, the load supplies reactive power |
    | resistive | $50$ | $4.6$ | $1058 + j0$ | 1058.0 | 1058.0 | $Q = 0$: current in phase with the voltage |

    `abs(S)` and `sqrt(P**2 + Q**2)` agree to the last digit because they are the same number: $|S|^2 = S S^* =
    (P + jQ)(P - jQ) = P^2 + Q^2$. The sign of $Q$ is the sign of $X$: $S = V I^* = |V|^2 / Z^* = |V|^2 (R + jX) /
    |Z|^2$, so $Q = |V|^2 X / |Z|^2$. The three loads here were chosen with the same $|Z| = 50\ \Omega$, which is
    why $|S| = 230^2 / 50 = 1058$ VA for all three while $P$ and $Q$ differ.

    ```octave title="three_loads.m"
    --8<-- "docs/part-1-math/ch01-complex-numbers/src/three_loads.m"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch01-complex-numbers/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 1

Numerator first: $(3 + 4j)(1 - j) = 3 - 3j + 4j - 4j^2 = 3 + j + 4 = 7 + j$. Dividing by $2j$ means multiplying
by $\dfrac{1}{2j} = \dfrac{-j}{2}$:

$$
\frac{7 + j}{2j} = \frac{(7 + j)(-j)}{2} = \frac{-7j - j^2}{2} = \frac{1 - 7j}{2} = 0.5 - 3.5j .
$$

Polar form: $|z| = \sqrt{0.5^2 + 3.5^2} = \sqrt{12.5} = 3.536$ and $\arg z = \operatorname{atan2}(-3.5, 0.5) =
-81.87°$ (fourth quadrant: real part positive, imaginary part negative). The polar route gives the same result and
is shorter: $|3 + 4j| = 5$, $|1 - j| = \sqrt 2$, $|2j| = 2$, so $|z| = 5\sqrt 2 / 2 = 3.536$; the arguments are
$53.13°$, $-45°$ and $90°$, so $\arg z = 53.13° - 45° - 90° = -81.87°$. `complex_plane.exercise1()` prints
`(0.5-3.5j)` and `polar_deg` gives $(3.5355, -81.87°)$.

## Exercise 2

For $z = -1 - j$, `math.atan(y / x)` computes $\arctan(1) = 45°$ and `math.atan2(y, x)` gives $-135°$
(`complex_plane.atan_vs_atan2(-1 - 1j)`). The second one is the argument: $-1 - j = \sqrt 2\, e^{-j3\pi/4}$ lies in
the third quadrant, and $45°$ points at $1 + j$, the opposite vector. The quotient $y/x$ loses the signs of $x$
and $y$ separately: $(-1)/(-1) = 1/1$, so $\arctan(y/x)$ cannot tell $z$ from $-z$ and always answers in
$(-90°, 90°)$, the right half plane. `atan2(y, x)` keeps both signs, returns a value in $(-180°, 180°]$ and handles
$x = 0$ without dividing by zero. This is why `cmath.phase`, Octave's `angle` and ToOp's
`switch_preparation.py` all use the two-argument form.

## Exercise 3

**Geometrically.** In GeoGebra, `a = cos(120°) + i sin(120°)`, `a^2` and `1` are three points on the unit circle
$120°$ apart; drawing them as vectors from the origin and adding them head to tail (`Vector(1) + Vector(a) +
Vector(a^2)`, or `1 + a + a^2` directly) returns to the origin: the three arrows form an equilateral triangle, so
the sum is the zero vector.

**Algebraically.** $a^3 = e^{j2\pi} = 1$, so $a$ is a root of $z^3 - 1 = (z - 1)(z^2 + z + 1)$. Since $a \ne 1$,
the other factor must vanish: $1 + a + a^2 = 0$. In components, $a = -\tfrac12 + j\tfrac{\sqrt 3}{2}$ and
$a^2 = e^{j4\pi/3} = -\tfrac12 - j\tfrac{\sqrt 3}{2}$, so the real parts give $1 - \tfrac12 - \tfrac12 = 0$ and
the imaginary parts cancel. The code prints $1 + a + a^2 = -1.1 \times 10^{-16} + 3.3 \times 10^{-16} j$, zero to
double precision. The same identity makes a balanced three-phase set carry no neutral current
([Chapter 10](../../part-2-circuits/ch10-three-phase-per-unit/index.md)).

## Exercise 4

The tapped winding adds the phasor $\Delta u\, e^{j\varphi}$ to the nominal $V_n$ (taken as the reference,
angle 0), so the output voltage is $V_n + \Delta u\, e^{j\varphi} = V_n \left(1 + \tfrac{\Delta u}{V_n}
e^{j\varphi}\right)$. Multiplying by the real positive $V_n$ does not change an argument, so the phase shift is
$\arg\!\left(1 + \tfrac{\Delta u}{V_n} e^{j\varphi}\right)$. Writing the number in rectangular form, its real part is
$V_n + \Delta u \cos\varphi$ and its imaginary part $\Delta u \sin\varphi$ (after multiplying both by $V_n$,
which again leaves the argument alone), and the argument of a complex number is `atan2(imaginary, real)`:

$$
\arg\!\left(1 + \frac{\Delta u}{V_n} e^{j\varphi}\right) = \operatorname{atan2}(\Delta u \sin\varphi,\ V_n + \Delta u \cos\varphi).
$$

With $V_n = 400$ kV, $n = 5$ and $s = 1\%$: $\Delta u = 400 \times 5 \times 1 / 100 = 20$ kV. For $\varphi = 90°$,
$\operatorname{atan2}(20, 400) = 2.862°$; for $\varphi = 60°$, $\operatorname{atan2}(20 \times 0.866, 400 + 10) =
\operatorname{atan2}(17.32, 410) = 2.419°$. `complex_plane.tap_phase_shift(400, 5, 1, φ)` returns both forms and
they agree to $10^{-12}$ degrees: $(2.8624°, 2.8624°)$ and $(2.4190°, 2.4190°)$. The quadrature booster
($\varphi = 90°$) shifts most per volt added; at $60°$ part of $\Delta u$ goes into raising the magnitude instead.

```octave title="tap_phase_shift.m"
--8<-- "docs/part-1-math/ch01-complex-numbers/src/tap_phase_shift.m"
```

[Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-1-math/ch01-complex-numbers/src){ target="_blank" } to see or download all the files of this chapter.

## Exercise 5

With sliders `t` and `ω`, the point `u = cos(ω t) + i sin(ω t)` $= e^{j\omega t}$ moves on the unit circle, and
the traced point `(t, x(u))` draws $x = \cos(\omega t)$: a **cosine**, the projection of the rotating unit phasor
on the real axis. `complex_plane.rotating_phasor(2π, t)` gives $\operatorname{Re}\, u = 1,\ 0.7071,\ 0,\ -1$ at
$t = 0,\ 0.125,\ 0.25,\ 0.5$, the same as `cos(2π t)`. $\omega$ is the angular speed of the phasor: it fixes how
far the arrow turns per unit time, so the trace repeats every $T = 2\pi/\omega$ and the cosine has frequency
$f = \omega / 2\pi$ (in a 50 Hz grid, $\omega = 100\pi$ rad/s). Scaling $u$ by a modulus $|V|$ and starting it at
an angle $\theta$ gives $|V| \cos(\omega t + \theta)$, which is how [Chapter 9](../../part-2-circuits/ch09-ac-steady-state/index.md)
turns every sinusoidal voltage into the fixed phasor $|V| e^{j\theta}$ and drops the common rotation
$e^{j\omega t}$.
