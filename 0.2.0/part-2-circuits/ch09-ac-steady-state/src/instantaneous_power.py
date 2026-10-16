"""Instantaneous power p(t) = v(t) i(t) of a single-phase load, sampled over one period (Chapter 9).

With v = √2 V cos(ωt) and i = √2 I cos(ωt − φ), the product is V I cos φ + V I cos(2ωt − φ): a constant P and
an oscillation at twice the supply frequency whose amplitude is the apparent power S = V I.
"""

import numpy as np


def sample_power(v_rms: float, i_rms: float, phi_deg: float, f_hz: float = 50.0, n: int = 2000) -> tuple:
    """Time axis over one period and the sampled v(t), i(t) and p(t)."""
    t = np.linspace(0.0, 1 / f_hz, n, endpoint=False)
    w = 2 * np.pi * f_hz
    v = np.sqrt(2) * v_rms * np.cos(w * t)
    i = np.sqrt(2) * i_rms * np.cos(w * t - np.radians(phi_deg))
    return t, v, i, v * i


def mean_and_swing(p: np.ndarray) -> tuple[float, float]:
    """Mean of p(t) over the period and the amplitude of its oscillation, (max − min) / 2."""
    return float(p.mean()), float((p.max() - p.min()) / 2)


def main() -> None:
    """Print the numbers of Exercise 3: P = V I cos φ and oscillation amplitude S = V I."""
    v_rms, i_rms, phi = 230.0, 10.0, 36.87
    _, _, _, p = sample_power(v_rms, i_rms, phi)
    mean, swing = mean_and_swing(p)
    print(f'mean of p(t) = {mean:.1f} W, V I cos(phi) = {v_rms * i_rms * np.cos(np.radians(phi)):.1f} W')
    print(f'oscillation amplitude = {swing:.1f} VA, V I = {v_rms * i_rms:.1f} VA')
    print(f'minimum of p(t) = {p.min():.1f} W (negative: energy returns to the source twice per period)')


if __name__ == '__main__':
    main()
