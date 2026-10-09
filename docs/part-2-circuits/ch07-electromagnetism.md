# Chapter 7. Electromagnetism refresher

**Weeks:** 1.5 · **Semester:** 2 · **Prerequisites:** none

!!! info "Why it matters for ToOp"

    Voltage, current, resistance, inductance and induction are the vocabulary of
    every later chapter; induction explains transformers and phase shifters. The vocabulary shows up directly
    in ToOp's code: branch limits are current ratings in amperes that are converted to power
    (`dc_solver/preprocess/powsybl/powsybl_helpers.py`, see
    [Chapter 10](ch10-three-phase-per-unit.md)), and the cascade relay model computes an impedance as voltage
    over current, Ohm's law in its complex form
    (`contingency/pandapower/cascade/detection/switch_preparation.py`).

## Topics

- Charge, electric field, potential; current and resistance; Ohm's law $V = RI$; power $P = VI = RI^2$;
  resistance of a conductor $R = \rho L / A$.
- Capacitance; magnetic fields and forces; Faraday's law $e = -\,d\Phi/dt$; inductance; energy in fields,
  $\tfrac12 L I^2$ and $\tfrac12 C V^2$.
- AC generation by induction; sinusoids, peak and RMS values (used in [Chapter 9](ch09-ac-steady-state.md)).

## Learning outcomes

- Compute the resistance and Joule losses of a transmission conductor and explain why power is transmitted at
  high voltage.
- Derive the sinusoidal EMF of a coil rotating in a uniform field from Faraday's law and compute its peak and
  RMS values.
- Compute the energy stored in an inductor and a capacitor and state which physical quantity cannot jump in
  each.

## Resources

- Courses: MIT OCW 8.02 (Spring 2019 or 2007) <https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2019/>.
- Free text: OpenStax *University Physics Vol. 2*, Ch. 7–10 (potential, capacitance, current, DC circuits)
  and Ch. 13–15 (induction, inductance, AC circuits): <https://openstax.org/details/books/university-physics-volume-2>.
- Textbooks:
  - Young & Freedman, *Sears & Zemansky's University Physics* (2 syllabi), Ch. 21–31. 14th ed. **BORROW**
    [`searszemanskysun0000youn`](https://archive.org/details/searszemanskysun0000youn).
    ES: *Física universitaria* (Pearson, 13th ed. 2014).
  - Serway & Jewett, *Physics for Scientists and Engineers* (2 syllabi). Vol. 2 (2013) **BORROW**
    [`isbn_9781490205649`](https://archive.org/details/isbn_9781490205649).
    ES: *Física para ciencias e ingeniería* (Cengage; 7th ed. cited at U. de Chile).
  - Tipler & Mosca, *Physics for Scientists and Engineers* (2 syllabi). **PRINT-DISABLED**.
    ES: *Física para la ciencia y la tecnología* (Reverté; 6th ed. required at UPC), vol. 2 **USER-UPLOAD (ES)** [`fisica-para-la-ciencia-y-la-tecnologia-tipler-6-edicion-vol-2-a`](https://archive.org/details/fisica-para-la-ciencia-y-la-tecnologia-tipler-6-edicion-vol-2-a).
- Documentation and tools: GeoGebra Classic <https://www.geogebra.org/classic>; matplotlib
  <https://matplotlib.org/stable/>.

## Lab

!!! example "Lab: from a rotating coil to RMS values"

    Tools: GeoGebra Classic, then Jupyter + NumPy + matplotlib.

    1. In GeoGebra, create sliders `ω` and `t`, draw a unit vector at angle `ω t` (the coil normal) and the point
       `(t, cos(ω t))` with trace on. The flux through a flat coil of area $A$ in a uniform field $B$ is
       $\Phi = BA\cos\omega t$; add the point `(t, ω sin(ω t))` for the normalized EMF and observe the 90° shift
       between flux and EMF.
    2. In a notebook, for a coil with $N = 100$ turns, $A = 0.01\ \text{m}^2$, $B = 0.5$ T at 50 Hz, sample one
       period, compute $\Phi(t)$ and $e(t) = -N\, d\Phi/dt$ with `np.gradient`, and compare with the analytic
       $e(t) = NBA\omega \sin\omega t$. Plot both with matplotlib.
    3. Compute the RMS of the sampled EMF as `np.sqrt(np.mean(e**2))` over whole periods and compare with
       $E_{\text{peak}}/\sqrt{2}$. Repeat for a signal with a 150 Hz component added and check that the squares of
       the component RMS values add.
    4. Plot the instantaneous power $p(t) = e(t)^2 / R$ into a resistor and show that its mean equals
       $E_{\text{RMS}}^2/R$: this is why RMS values are used for AC power.

## Exercises

1. An aluminium conductor ($\rho \approx 2.82\cdot10^{-8}\ \Omega\,\text{m}$) is 100 km long with a
   500 mm² cross-section. Compute its resistance and the Joule losses at 1 000 A. *(Pen and paper)*
   *Answer:* $5.64\ \Omega$; $5.64$ MW.
2. A two-conductor line made of the conductor of Exercise 1 delivers 200 MW. Compare the current and the
   losses at 200 kV and at 400 kV, and state the scaling law. *(Pen and paper)* *Answer:* 1 000 A and
   11.3 MW vs 500 A and 2.8 MW; losses scale as $1/V^2$.
3. For the coil of the lab, compute the peak and RMS EMF by hand. *(Pen and paper)* *Answer:* $157.1$ V and
   $111.1$ V.
4. Compute the energy stored in a 1 H inductor carrying 1 000 A and in a 1 µF capacitor at 230.9 kV. Explain
   why the current through an inductor and the voltage across a capacitor cannot change instantaneously.
   *(Pen and paper)*
5. Sample $v(t) = 325 \sin(2\pi 50 t)$ and verify numerically that its RMS is about 230 V. How many samples per
   period do you need for 0.1% accuracy if the sampling window is not an integer number of periods?
   *(Jupyter: NumPy + matplotlib)*
