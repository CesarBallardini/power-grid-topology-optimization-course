# Chapter 1. Complex numbers and phasor arithmetic

**Weeks:** 1 · **Semester:** 1 · **Prerequisites:** none

!!! info "Why it matters for ToOp"

    AC power flow validation works with voltages $V = |V|e^{j\theta}$, complex power $S = V I^*$ and admittances
    $Y = G + jB$. The distance protection model converts P and Q to an impedance angle with `np.arctan2`
    (`contingency/pandapower/cascade/detection/switch_preparation.py`), and the voltage-angle-difference results
    compute transformer phase shifts from tap position, tap step and step angle with `atan2`
    (`contingency/pandapower/pandapower_helpers/results/va_diff_results.py`). Both are complex-number arguments
    in disguise.

## Topics

- Algebra of complex numbers, conjugate, modulus.
- Polar and exponential form; Euler's formula $e^{j\theta} = \cos\theta + j\sin\theta$; `atan2` and quadrants.
- Multiplication and division as rotation and scaling; roots of unity (preview of three-phase, revisited in
  [Chapter 10](../part-2-circuits/ch10-three-phase-per-unit.md) as the operator $a = e^{j2\pi/3}$).
- Rotating phasors $V e^{j\omega t}$ and their projection on the real axis (used in
  [Chapter 9](../part-2-circuits/ch09-ac-steady-state.md)).

## Learning outcomes

- Convert fluently between rectangular and polar forms and compute products, quotients and powers.
- Explain why a phase shift is a multiplication by $e^{j\theta}$ and show it in GeoGebra.
- Use `atan2` to recover an angle in any quadrant and explain when $\arctan(y/x)$ gives the wrong answer.
- Compute complex power $S = V I^*$ for a simple load and check that $|S| = \sqrt{P^2 + Q^2}$.

## Resources

- Courses and videos: Khan Academy complex numbers
  (<https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:complex>); Welch Labs, "Imaginary Numbers
  Are Real" (<https://www.youtube.com/playlist?list=PLiaHhY2iBX9g6KIvZ_703G3KJXapKkNaF>); MIT OCW 18.04,
  Topic 1 (<https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/>).
- Free texts: OpenStax *Precalculus 2e* §8.5
  (<https://openstax.org/books/precalculus-2e/pages/8-5-polar-form-of-complex-numbers>); Beck, Marchesi,
  Pixton & Sabalka, *A First Course in Complex Analysis*, Ch. 1 (FREE, <https://matthbeck.github.io/complex.html>).
- Textbooks:
  - Brown & Churchill, *Complex Variables and Applications* (4 syllabi), Ch. 1. 8th ed. 2008 **BORROW**
    [`complexvariables0000jame`](https://archive.org/details/complexvariables0000jame).
    ES: *Variable compleja y aplicaciones* (McGraw-Hill Interamericana, 7th ed. 2004).
  - Spiegel, *Schaum's Outline of Complex Variables*, Ch. 1. 1964 **BORROW**
    [`schaumsoutlineof0000spie_b4h6`](https://archive.org/details/schaumsoutlineof0000spie_b4h6).
    ES: *Variable compleja* (Schaum, McGraw-Hill).
- Documentation and tools:
  - GeoGebra Classic (free, runs in the browser): <https://www.geogebra.org/classic>; manual page
    "Complex Numbers" (`abs`, `arg`, `conjugate`): <https://geogebra.github.io/docs/manual/en/Complex_Numbers/>.
  - Python `cmath` module: <https://docs.python.org/3/library/cmath.html>.
  - GNU Octave, "Complex Arithmetic" (`abs`, `angle`, `conj`): <https://docs.octave.org/latest/Complex-Arithmetic.html>.

## Worked example

A load $Z = 40 + j30\ \Omega$ is fed by $V = 230\angle 0^\circ$ V. Then $|Z| = 50\ \Omega$, $\angle Z = 36.87^\circ$
and

$$
I = \frac{V}{Z} = 4.6\angle -36.87^\circ\ \text{A} = 3.68 - j2.76\ \text{A},
\qquad
S = V I^* = 1058\angle 36.87^\circ\ \text{VA} = 846.4 + j634.8\ \text{VA}.
$$

The current lags the voltage by the impedance angle, and $P = 846.4$ W, $Q = 634.8$ var. In plain Python (no
libraries) and in Octave:

```python
import cmath, math

V, Z = 230, 40 + 30j
I = V / Z
S = V * I.conjugate()
print(abs(I), math.degrees(cmath.phase(I)))   # 4.6  -36.87
print(S, abs(S), math.hypot(S.real, S.imag))  # (846.4+634.8j)  1058.0  1058.0
```

```octave
V = 230; Z = 40 + 30i;
I = V / Z;
S = V * conj(I)
[abs(I), angle(I) * 180 / pi]
[abs(S), hypot(real(S), imag(S))]
```

## Lab

!!! example "Lab: rotations in the complex plane"

    Tools: GeoGebra Classic, then a Jupyter notebook with plain Python (`cmath`) or Octave. No NumPy is needed
    yet; [Chapter 30](../part-5-computing/ch30-scientific-python.md) starts in parallel.

    1. In GeoGebra, enter `z = 3 + 4i` and a slider `θ = Slider(0°, 360°, 1°)`. Define
       `w = z * (cos(θ) + i sin(θ))`. Move the slider and record `abs(z)`, `abs(w)`, `arg(z)` and `arg(w)`:
       the modulus stays at 5 and the argument grows by θ.
    2. Add `u = 1 + 0.5i` and `p = z * u`. Check that `abs(p) = abs(z) abs(u)` and `arg(p) = arg(z) + arg(u)`
       (modulo 360°). Explain in one sentence why multiplication is "rotate and scale".
    3. Draw the three cube roots of unity with `Sequence(cos(k 120°) + i sin(k 120°), k, 0, 2)` and check that
       their sum is 0. This is the balanced three-phase set of
       [Chapter 10](../part-2-circuits/ch10-three-phase-per-unit.md).
    4. In the notebook (or Octave), repeat the worked example for $Z = R + jX$ with three loads of your choice
       (inductive, capacitive, purely resistive). For each, verify `S = V * conj(I)` and compare `abs(S)` with
       `sqrt(P**2 + Q**2)`. State the sign of Q for each load.

## Exercises

1. By hand, compute $\dfrac{(3 + 4j)(1 - j)}{2j}$ in rectangular and polar form. *Answer:* $0.5 - j3.5 =
   3.536\angle -81.87^\circ$.
2. For $z = -1 - j$, compare $\arctan(y/x)$ with `atan2(y, x)`. Which one gives the argument of $z$, and why?
   *(Jupyter: `math.atan`, `math.atan2`)* *Hint:* one returns $45^\circ$, the other $-135^\circ$.
3. Show that $1 + a + a^2 = 0$ for $a = e^{j2\pi/3}$, first geometrically in GeoGebra (vector sum of the three
   arrows), then algebraically. *(GeoGebra)*
4. A transformer adds a voltage $\Delta u = V_n \cdot n \cdot s/100$ at step angle $\varphi$ to its nominal
   voltage $V_n$, where $n$ is the number of tap steps from neutral and $s$ the step in percent. Show that the
   resulting phase shift is $\arg\left(1 + \frac{\Delta u}{V_n} e^{j\varphi}\right) = \operatorname{atan2}(\Delta u
   \sin\varphi,\ V_n + \Delta u \cos\varphi)$, the formula in ToOp's
   `contingency/pandapower/pandapower_helpers/results/va_diff_results.py`. Evaluate it for $V_n = 400$ kV,
   $n = 5$, $s = 1\%$ and $\varphi = 90^\circ$ and $60^\circ$. *(Pen and paper, check in Octave or Python)*
   *Answer:* $2.86^\circ$ and $2.42^\circ$.
5. Draw a rotating phasor in GeoGebra: sliders `t` and `ω`, point `u = cos(ω t) + i sin(ω t)`, and the point
   `(t, x(u))` with trace on. Which familiar function do you get, and what is the role of $\omega$?
   *(GeoGebra)*
