# Chapter 7 – Solutions

Solutions of the Lab and the Exercises of [Chapter 7](index.md). Try each task before reading its solution.
The code is in the chapter's `src` folder: [`coil_emf.py`](src/coil_emf.py) (the rotating coil, RMS values and
the sampling study; `python coil_emf.py` prints every number of the Lab answers and of Exercises 3 and 5) and
[`conductor.py`](src/conductor.py) (`python conductor.py` prints those of Exercises 1, 2 and 4). The notebooks
[`examples.ipynb`](src/examples.ipynb), [`lab.ipynb`](src/lab.ipynb) and [`exercises.ipynb`](src/exercises.ipynb)
import these modules.

## Lab

1. **GeoGebra** (not run here; GeoGebra is interactive). Method: two sliders `ω` (say 1 to 10) and `t`
   (0 to 10, animated); the vector `Vector((0, 0), (cos(ω t), sin(ω t)))` is the coil normal, which makes an
   angle $\omega t$ with the field (taken along the $x$ axis); the point `(t, cos(ω t))` with *Trace on*
   draws the normalized flux $\Phi / (BA) = \cos\omega t$ as `t` runs, and `(t, ω sin(ω t))` draws the
   normalized EMF $e / (NBA) = -\,d(\cos\omega t)/dt = \omega \sin\omega t$. Expected picture: the EMF trace
   is zero where the flux trace is at a peak (the normal along the field) and largest where the flux crosses
   zero (the normal across the field, when the flux changes fastest). The EMF is shifted by a quarter period,
   $90^\circ$, with respect to the flux; raising `ω` raises the EMF amplitude in proportion and leaves the
   flux amplitude alone. The numeric check in `coil_emf.py`: $e(0) = 0$ where $\Phi(0) = 0.5$ Wb-turn is
   the maximum, and the mean of $\Phi(t)\, e(t)$ over a period is zero (the two are orthogonal).

2. **Flux and EMF of the coil.** With $N = 100$, $A = 0.01\ \text{m}^2$, $B = 0.5$ T and
   $\omega = 2\pi \cdot 50 = 314.16\ \text{rad/s}$ the flux linkage is $N\Phi = NBA\cos\omega t$ with peak
   $0.5$ Wb-turn, and $e = -N\, d\Phi/dt = NBA\omega \sin\omega t$ with peak $NBA\omega = 157.08$ V. Sampled
   at 1 000 points per period, `np.gradient` reproduces the analytic EMF within $0.49$ V ($3.1\cdot10^{-3}$ of
   the peak); the error is $1.6\cdot10^{-1}$ with 20 samples per period and $3.1\cdot10^{-2}$ with 100, so it
   falls as $1/n$: `np.gradient` is second order inside the window but first order at the two end points,
   where it uses one-sided differences, and the largest error sits there. Plotted together, the two EMF curves
   coincide and both are shifted by a quarter period with respect to the flux.

    ```python title="coil_emf.py: flux and EMF"
    --8<-- "docs/part-2-circuits/ch07-electromagnetism/src/coil_emf.py:emf"
    ```

    [Open the `src/` folder on GitHub](https://github.com/CesarBallardini/power-grid-topology-optimization-course/tree/main/docs/part-2-circuits/ch07-electromagnetism/src){ target="_blank" } to see or download all the files of this chapter.

3. **RMS.** Over one period (and over three) the sampled RMS of the EMF is $111.072$ V $= 157.08 / \sqrt2$. A
   150 Hz component $40 \sin 3\omega t$ has RMS $28.284$ V $= 40/\sqrt2$; the RMS of the sum is $114.617$ V,
   and $\sqrt{111.072^2 + 28.284^2} = 114.617$ V: the squares add. They add because the cross term
   $2\,\overline{e_1 e_3}$ vanishes, sinusoids of different frequencies being orthogonal over a whole number of
   periods of both (here one period of the fundamental holds three of the harmonic).

4. **Power into a resistor.** With $R = 100\ \Omega$ the instantaneous power $p(t) = e^2/R$ swings between 0
   and $246.74$ W ($= E_\text{peak}^2/R$) at 100 Hz, twice the frequency of the EMF, and its mean over a
   period is $123.37$ W, exactly $E_\text{RMS}^2/R = 111.072^2 / 100$. That is the definition of the RMS
   value: the DC voltage that would heat the same resistor at the same average rate. A DC calculation with
   $E_\text{RMS}$ gives the AC average power, which is why ratings and bills are in RMS volts and amperes.

    ```python title="coil_emf.py: RMS and mean power"
    --8<-- "docs/part-2-circuits/ch07-electromagnetism/src/coil_emf.py:rms"
    ```

## Exercise 1

$R = \rho L / A = 2.82\cdot10^{-8} \times 10^5 / (500\cdot10^{-6}) = 5.64\ \Omega$. At $I = 1\,000$ A the
Joule losses are $P = RI^2 = 5.64 \times 10^6\ \text{W} = 5.64$ MW, in one conductor (`conductor.py` prints
both numbers). Whether 5.64 MW is a lot depends on the power the conductor carries, which depends on the
voltage: that is Exercise 2.

```python title="conductor.py: losses"
--8<-- "docs/part-2-circuits/ch07-electromagnetism/src/conductor.py:losses"
```

## Exercise 2

The current that carries $P = 200$ MW at voltage $V$ is $I = P/V$: $1\,000$ A at 200 kV and $500$ A at
400 kV. Each of the two conductors dissipates $RI^2$, so the line loses $2RI^2 = 2 \times 5.64 \times 10^6 =
11.28$ MW (5.64 % of the power delivered) at 200 kV and $2 \times 5.64 \times 500^2 = 2.82$ MW (1.41 %) at
400 kV. Substituting $I = P/V$ gives the scaling law $P_\text{loss} = 2R P^2 / V^2$: for a fixed power and
conductor the losses fall as $1/V^2$, so doubling the voltage divides them by four. The same law holds for
any number of conductors and is the reason transmission runs at hundreds of kilovolts, with transformers
(induction, the topic of this chapter) to step the voltage up and down.

## Exercise 3

$E_\text{peak} = NBA\omega = 100 \times 0.5 \times 0.01 \times 2\pi \times 50 = 0.5 \times 314.16 = 157.1$ V
and $E_\text{RMS} = E_\text{peak}/\sqrt2 = 111.1$ V. The sampled values of the lab are $157.08$ V and
$111.07$ V.

## Exercise 4

$W_L = \tfrac12 L I^2 = \tfrac12 \times 1 \times 1000^2 = 500$ kJ and
$W_C = \tfrac12 C V^2 = \tfrac12 \times 10^{-6} \times (230.9\cdot10^3)^2 = 26.66$ kJ. The energy of an
inductor is a function of its current and that of a capacitor a function of its voltage. A jump in the current
of an inductor would be a jump in its stored energy, which needs an infinite power $p = v\,i$ and so an
infinite voltage $v = L\, di/dt$; a jump in the voltage of a capacitor would need an infinite current
$i = C\, dv/dt$. Neither exists in a real circuit, so $i_L$ and $v_C$ are continuous: they are the state
variables of the circuit, and whatever a switch does, they keep their value across the switching instant. This
is why opening a breaker on an inductive line produces an arc (the current insists on continuing) and why a
capacitor bank is pre-charged or switched at a voltage zero.

## Exercise 5

Sampling $v(t) = 325 \sin(2\pi 50 t)$ at 1 000 points over one period gives an RMS of $229.81$ V
$= 325/\sqrt2$, the nominal 230 V of a European outlet. The number of samples per period is not what limits
the accuracy. Over a whole number of periods the discrete RMS
of a sinusoid is exact already with 3 samples per period (the error is $10^{-16}$ with 3, 4, 10 or 100
samples per period over 10 periods), because the discrete mean of $\sin^2$ over an integer number of
equally spaced samples per period is exactly $1/2$ when there are at least three. Over a window that is not a
whole number of periods the error is a window effect: the mean of $\sin^2\omega t$ over $[0, T]$ is
$\tfrac12 - \sin(2\omega T) / (4\omega T)$, so the relative error of the RMS is at most $1/(4\omega T) =
1/(8\pi n)$ for a window of $n$ periods, whatever the sampling. With 1 000 samples per period the error is
$2.7\cdot10^{-2}$ over 1.4 periods, $3.6\cdot10^{-3}$ over 10.4, $9.3\cdot10^{-4}$ over 40.4 and
$3.8\cdot10^{-4}$ over 100.4 periods, each just below its bound ($2.8\cdot10^{-2}$, $3.8\cdot10^{-3}$,
$9.8\cdot10^{-4}$, $4.0\cdot10^{-4}$), and over 10.4 periods it is $1.5\cdot10^{-3}$ with 10 samples per
period, $3.5\cdot10^{-3}$ with 100 and $3.6\cdot10^{-3}$ with 1 000: more samples make the discrete sum a
better approximation of the continuous integral, which is itself off by the window term. So the answer is:
for 0.1 % accuracy with a window of arbitrary length you need the window to hold at least
$1/(8\pi \cdot 0.001) \approx 40$ periods (0.8 s at 50 Hz), and a few samples per period are then enough; or
you synchronise the window to whole periods, as power meters do, and 3 samples per period are exact.
`rms_window_error` and `rms_window_error_bound` in `coil_emf.py` give the table.
