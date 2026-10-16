"""Phasor arithmetic of phase-shifting transformers (Chapter 11): quadrature booster, symmetric PST, general step.

A quadrature booster adds to the input voltage U a series voltage k j U, perpendicular to it: W_q = U (1 + jk),
with |W_q| = |U| √(1 + k²) and a shift atan(k). A symmetric PST rotates U by α without changing its magnitude:
W_s = U e^{jα}. A general step adds k e^{jφ} U at angle φ, with the shift atan2(k sin φ, 1 + k cos φ) of
Chapter 1, Exercise 4.
"""

import cmath
import math


def quadrature_booster(u: complex, k: float) -> complex:
    """Output voltage U + k j U of a quadrature booster with relative series voltage ``k``."""
    return u + 1j * k * u


def symmetric_pst(u: complex, alpha_deg: float) -> complex:
    """Output voltage U e^{jα} of a symmetric PST shifting by ``alpha_deg`` degrees."""
    return u * cmath.exp(1j * math.radians(alpha_deg))


def series_voltage(u: complex, k: float, phi_deg: float) -> complex:
    """Output voltage U + k e^{jφ} U of a general tap step of relative size ``k`` at step angle ``phi_deg``."""
    return u * (1 + k * cmath.exp(1j * math.radians(phi_deg)))


def phase_shift_deg(k: float, phi_deg: float) -> float:
    """The shift atan2(k sin φ, 1 + k cos φ) of a step of relative size ``k`` at angle ``phi_deg``, in degrees."""
    phi = math.radians(phi_deg)
    return math.degrees(math.atan2(k * math.sin(phi), 1 + k * math.cos(phi)))


def quadrature_magnitude_for_shift(alpha_deg: float) -> float:
    """Output magnitude of a quadrature booster producing the shift ``alpha_deg``: 1 / cos α, since k = tan α."""
    return 1 / math.cos(math.radians(alpha_deg))


def main() -> None:
    """Print the numbers of lab steps 1 and 2 and of Exercise 5."""
    u = 1 + 0j
    for k in (0.05, 0.1, 0.2):
        w_q = quadrature_booster(u, k)
        alpha = math.degrees(cmath.phase(w_q))
        w_s = symmetric_pst(u, alpha)
        print(f'k = {k:.2f}: shift {alpha:.3f} deg, |W_q| = {abs(w_q):.4f}, |W_s| = {abs(w_s):.4f}')
    for phi in (90.0, 60.0):
        w_g = series_voltage(u, 0.05, phi)
        print(
            f'k = 0.05 at phi = {phi:.0f} deg: arg(W_g) = {math.degrees(cmath.phase(w_g)):.3f} deg, '
            f'formula {phase_shift_deg(0.05, phi):.3f} deg, |W_g| = {abs(w_g):.4f}'
        )
    for alpha in (5.0, 10.0, 20.0):
        print(
            f'exercise 5: same shift {alpha:.0f} deg: symmetric |W| = 1.0000, quadrature |W| = '
            f'{quadrature_magnitude_for_shift(alpha):.4f}'
        )


if __name__ == '__main__':
    main()
