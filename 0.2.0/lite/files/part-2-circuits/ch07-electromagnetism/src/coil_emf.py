"""A coil rotating in a uniform field: flux, induced EMF, RMS values and power into a resistor (Chapter 7 lab).

The coil has N turns of area A in a field B and turns at f hertz, so Φ(t) = N B A cos(ωt) and
e(t) = −dΦ/dt = N B A ω sin(ωt). Exercise 5 uses the same sampling tools on v(t) = 325 sin(2 pi 50 t).
"""

import numpy as np

N_TURNS, AREA, B_FIELD, FREQ = 100, 0.01, 0.5, 50.0  # turns, m², T, Hz
OMEGA = 2 * np.pi * FREQ
E_PEAK = N_TURNS * B_FIELD * AREA * OMEGA  # 157.08 V


# --8<-- [start:emf]
def sample_period(samples: int = 1000, periods: float = 1.0, freq: float = FREQ) -> np.ndarray:
    """``samples`` instants per period over ``periods`` periods, the end point excluded (t = 0, Δt, 2Δt, ...)."""
    return np.arange(round(samples * periods)) / (samples * freq)


def flux(t: np.ndarray) -> np.ndarray:
    """Flux linkage N B A cos(ωt) in weber-turns."""
    return N_TURNS * B_FIELD * AREA * np.cos(OMEGA * t)


def emf_numeric(t: np.ndarray) -> np.ndarray:
    """The EMF e = −dΦ/dt by central differences (``np.gradient``) on the sampled flux."""
    return -np.gradient(flux(t), t)


def emf_analytic(t: np.ndarray) -> np.ndarray:
    """The EMF e = N B A ω sin(ωt)."""
    return E_PEAK * np.sin(OMEGA * t)


# --8<-- [end:emf]


# --8<-- [start:rms]
def rms(x: np.ndarray) -> float:
    """Root mean square of the samples."""
    return float(np.sqrt(np.mean(x**2)))


def mean_power(e: np.ndarray, r: float) -> float:
    """Average of p(t) = e(t)² / R over the samples, in watts."""
    return float(np.mean(e**2 / r))


# --8<-- [end:rms]


def gradient_error(samples: int) -> float:
    """Largest |numeric − analytic| EMF over one period, relative to the peak, for ``samples`` per period."""
    t = sample_period(samples)
    return float(np.max(np.abs(emf_numeric(t) - emf_analytic(t))) / E_PEAK)


def rms_window_error(periods: float, samples: int = 1000, amplitude: float = 325.0) -> float:
    """Relative error of the sampled RMS of a sinusoid over a window of ``periods`` periods (Exercise 5)."""
    t = sample_period(samples, periods)
    return rms(amplitude * np.sin(OMEGA * t)) / (amplitude / np.sqrt(2)) - 1


def rms_window_error_bound(periods: float) -> float:
    """Worst-case relative RMS error for a window of ``periods`` periods: 1 / (8π · periods), from the leakage term.

    The mean of sin² over [0, T] is 1/2 − sin(2ωT)/(4ωT); its relative error is at most 1/(2ωT), and the square
    root halves it: 1/(4ωT) = 1/(8π · T/T₀).
    """
    return 1 / (8 * np.pi * periods)


def periods_for_accuracy(rel: float) -> float:
    """Window length, in periods, that keeps the worst-case RMS error of a sinusoid below ``rel``: 1 / (8π rel)."""
    return 1 / (8 * np.pi * rel)


def main() -> None:
    """Print the numbers quoted in the lab solution and in Exercises 3 and 5."""
    t = sample_period(1000)
    e_num, e_ana = emf_numeric(t), emf_analytic(t)
    print(f'peak EMF N B A w = {E_PEAK:.2f} V, RMS = {E_PEAK / np.sqrt(2):.2f} V')
    print(f'flux peak {flux(t).max():.3f} Wb-turn; EMF lags/leads: e = 0 where the flux is extreme, e.g. t = 0')
    print(f'np.gradient vs analytic, 1000 samples/period: max |error| = {np.max(np.abs(e_num - e_ana)):.4f} V')
    print(
        '  relative error for 20, 100, 1000 samples per period:', [f'{gradient_error(n):.1e}' for n in (20, 100, 1000)]
    )
    e3p = emf_analytic(sample_period(1000, 3))
    print(f'sampled RMS over 1 period: {rms(e_ana):.3f} V, over 3 periods: {rms(e3p):.3f} V')
    e3 = 40.0 * np.sin(3 * OMEGA * t)  # a 150 Hz component of 40 V peak
    print(
        f'150 Hz component alone: RMS {rms(e3):.3f} V; sum: RMS {rms(e_ana + e3):.3f} V; '
        f'sqrt of sum of squares {np.hypot(rms(e_ana), rms(e3)):.3f} V'
    )
    r = 100.0
    print(
        f'mean power into {r:.0f} ohm: {mean_power(e_ana, r):.2f} W; E_rms^2/R = {rms(e_ana) ** 2 / r:.2f} W; '
        f'peak power {np.max(e_ana**2 / r):.2f} W'
    )
    print(f'Exercise 5: RMS of 325 sin(2 pi 50 t) over 1 period, 1000 samples: {rms(325 * np.sin(OMEGA * t)):.2f} V')
    print(
        '  integer window (10 periods), samples per period 3, 4, 10, 100:',
        [f'{rms_window_error(10, n, 325):.1e}' for n in (3, 4, 10, 100)],
    )
    print(
        '  window 10.4 periods, samples per period 10, 100, 1000:',
        [f'{rms_window_error(10.4, n, 325):.1e}' for n in (10, 100, 1000)],
    )
    print(
        '  1000 samples/period, window 1.4, 10.4, 40.4, 100.4 periods:',
        [f'{rms_window_error(p, 1000, 325):.1e}' for p in (1.4, 10.4, 40.4, 100.4)],
    )
    print(
        '  worst-case bound 1/(8 pi n) for n = 1.4, 10.4, 40.4, 100.4 periods:',
        [f'{rms_window_error_bound(p):.1e}' for p in (1.4, 10.4, 40.4, 100.4)],
    )
    print(f'  periods needed for 0.1 %: {periods_for_accuracy(1e-3):.1f}')


if __name__ == '__main__':
    main()
