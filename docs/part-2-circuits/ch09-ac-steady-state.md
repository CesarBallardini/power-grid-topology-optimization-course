# Chapter 9. AC steady state and complex power

**Weeks:** 2.5 · **Semester:** 2 · **Prerequisites:** [Chapter 1](../part-1-math/ch01-complex-numbers.md), [Chapter 8](ch08-dc-circuits.md)

!!! info "Why it matters for ToOp"

    AC validation reports active and reactive power, current magnitude, voltage
    magnitude and angle per element (`interfaces/` loadflow result schemas). DC limits are derived from current
    ratings assuming a power factor of 1 (`dc_power_factor` in `grid_helpers/powsybl/loadflow_parameters.py`).
    The DC model that ToOp optimizes on is the small-angle, flat-voltage limit of the transfer formula
    $P = V_1 V_2 \sin\delta / X$ derived in this chapter.

## Topics

- Sinusoids, RMS values, phasors; impedance and admittance.
- Nodal analysis with complex admittances (revisits [Chapter 8](ch08-dc-circuits.md) with $Y$ instead of $G$).
- Instantaneous, active, reactive and apparent power; complex power $S = V I^* = P + jQ$; power factor and its
  correction.
- Power transfer between two AC sources through a reactance,
  $P = \dfrac{V_1 V_2}{X}\sin\delta$ (the seed of DC power flow in
  [Chapter 16](../part-3-power-systems/ch16-dc-power-flow.md)).

## Learning outcomes

- Solve single-phase AC circuits with phasors and compute $P$, $Q$, $S$ and the power factor.
- Derive $P$ and $Q$ transferred through a lossless reactance and plot the $P$–$\delta$ curve.
- Quantify the error of the linearization $P \approx \delta / X$ (per unit) as a function of $\delta$.

## Resources

- Courses: Coursera *Linear Circuits 2: AC Analysis* (Georgia Tech)
  <https://www.coursera.org/learn/linear-circuits-ac-analysis>; MIT OCW 6.061 notes Ch. 1–2 (network theory,
  AC power flow in linear networks), FREE
  <https://ocw.mit.edu/courses/6-061-introduction-to-electric-power-systems-spring-2011/pages/readings/>.
- Free texts: Kuphaldt Vol. II AC <https://www.ibiblio.org/kuphaldt/electricCircuits/AC/>; Ulaby et al. (open
  access).
- Textbooks: the sinusoidal steady-state and AC power chapters of the circuit books in [Chapter 8](ch08-dc-circuits.md).
- Documentation and tools: GeoGebra Classic <https://www.geogebra.org/classic> and the "Complex Numbers"
  manual page <https://geogebra.github.io/docs/manual/en/Complex_Numbers/>; GNU Octave "Complex Arithmetic"
  <https://docs.octave.org/latest/Complex-Arithmetic.html>.

## Worked example

Bus 1 has voltage $V_1\angle\delta$, bus 2 has $V_2\angle 0$, and they are joined by a lossless reactance $jX$.
The current from 1 to 2 is $I = (V_1 e^{j\delta} - V_2)/(jX)$, and the complex power leaving bus 1 is

$$
S_{12} = V_1 e^{j\delta}\, I^* = \frac{V_1^2 - V_1 V_2 e^{j\delta}}{-jX}
= \underbrace{\frac{V_1 V_2}{X}\sin\delta}_{P_{12}} + j\,\underbrace{\frac{V_1^2 - V_1 V_2\cos\delta}{X}}_{Q_{12}} .
$$

Active power follows the angle difference; reactive power follows mainly the magnitude difference. With
$V_1 = V_2 = 1$ per unit and small $\delta$ (in radians), $\sin\delta \approx \delta$ gives the DC approximation

$$
P_{12} \approx \frac{\delta}{X}.
$$

## Lab

!!! example "Lab: the P–δ curve and the DC approximation"

    Tools: GeoGebra Classic, then Jupyter + NumPy + pandas + matplotlib.

    1. In GeoGebra, enter `p(x) = V_1 V_2 sin(x) / X_L` and accept the automatic sliders for `V_1`, `V_2` and
       `X_L` (or create them with the Slider tool); add `q(x) = x / X_L` for the linearization. Move the
       sliders and note the maximum transferable power and the angle where it occurs.
    2. Phasor diagram: points `U_1 = V_1 (cos(d) + i sin(d))` and `U_2 = V_2 + 0i` with a slider `d`, the
       difference `U_1 - U_2` and the current `I_L = (U_1 - U_2) / (X_L i)`. Check that `I_L` is perpendicular to
       `U_1 - U_2` and watch how its angle relative to `U_1` changes with `d`.
    3. In a notebook, tabulate with pandas, for $\delta = 0^\circ, 5^\circ, \dots, 90^\circ$ and $V_1 = V_2 = 1$ pu,
       $X = 0.1$ pu: the exact $P$, the DC estimate $\delta/X$ and the relative error. Plot the error with
       matplotlib. *Check:* about 0.5% at $10^\circ$, 4.7% at $30^\circ$ and 21% at $60^\circ$.
    4. Repeat with $V_1 = 1.05$ and $V_2 = 0.95$ pu and separate the error caused by the angle linearization from
       the error caused by assuming flat voltage magnitudes.

## Exercises

1. A series RL load, $R = 10\ \Omega$ and $L = 50$ mH, is fed by 230 V at 50 Hz. Compute $Z$, $I$, $S$ and the power
   factor, then the capacitance in parallel that raises the power factor to 0.95. *(Pen and paper, check in
   Octave)* *Answer:* $Z = 10 + j15.71\ \Omega$, $|I| = 12.35$ A, $S = 1526 + j2396$ VA, $\text{pf} = 0.537$,
   $C \approx 114\ \mu\text{F}$.
2. Rotating phasors: in GeoGebra, animate `u = V_m (cos(ω t + φ) + i sin(ω t + φ))` with sliders and trace
   `(t, x(u))`. Show that the phasor $V_m e^{j\varphi}$ is the value of the rotating vector at $t = 0$. *(GeoGebra)*
3. Sample $v(t) = \sqrt{2}\, V \cos\omega t$ and $i(t) = \sqrt{2}\, I \cos(\omega t - \varphi)$, compute
   $p(t) = v(t)\,i(t)$, and verify numerically that its mean is $P = VI\cos\varphi$ and its oscillation
   amplitude is $S = VI$. *(Jupyter: NumPy + matplotlib)*
4. From the worked example, show that for $\delta = 0$ reactive power flows from the bus with the higher voltage
   magnitude, and explain why the DC approximation ignores $Q$ altogether. *(Pen and paper)*
5. Solve a three-node AC circuit (two sources, an RL branch, an RC branch) with complex nodal analysis,
   `Y \ I` in Octave or `np.linalg.solve` in NumPy, and verify that the sum of complex powers injected equals
   the sum absorbed. *(Octave or Jupyter: NumPy)*
6. A branch limit is a current $I_{\max}$. If the DC model converts it to $P_{\max} = V I_{\max}$ (power factor 1,
   as with ToOp's `dc_power_factor=1.0`) but the branch operates at power factor 0.95, by how much does the DC
   limit overestimate the admissible active power? *(Pen and paper)* *Answer:* about 5.3%.
