"""Rotations, products, roots of unity and a transformer's phase shift with plain ``cmath`` (Chapter 1).

The numeric checks of the Lab (rotating ``z = 3 + 4i``, the product with ``u = 1 + 0.5i``, the cube roots of unity,
three loads on a 230 V source) and of the Exercises (the hand computation of Exercise 1, ``atan`` against ``atan2``,
the tap-changer phase shift of Exercise 4, the rotating phasor of Exercise 5). No NumPy: the ``complex`` type and
``cmath`` are enough. ``python complex_plane.py`` prints every number the solutions quote.
"""

import cmath
import math

from phasor_power import complex_power, load_current

LOADS = {'inductive': 40 + 30j, 'capacitive': 40 - 30j, 'resistive': 50 + 0j}  # ohm, Lab step 4


def polar_deg(z: complex) -> tuple[float, float]:
    """Modulus and argument in degrees, the argument from ``cmath.phase`` (an ``atan2``) in (-180, 180]."""
    return abs(z), math.degrees(cmath.phase(z))


def rotate(z: complex, theta_deg: float) -> complex:
    """``z`` turned by ``theta_deg`` degrees: the product with exp(j theta), modulus unchanged."""
    return z * cmath.exp(1j * math.radians(theta_deg))


def wrap_deg(angle: float) -> float:
    """The same angle brought into (-180, 180], so that sums of arguments compare with ``polar_deg``."""
    return (angle + 180.0) % 360.0 - 180.0 if (angle + 180.0) % 360.0 != 0 else 180.0


def product_check(z: complex, u: complex) -> dict[str, float]:
    """Modulus and argument of ``z * u`` next to the product of moduli and the sum of arguments (Lab step 2)."""
    mod_z, arg_z = polar_deg(z)
    mod_u, arg_u = polar_deg(u)
    mod_p, arg_p = polar_deg(z * u)
    return {'abs_p': mod_p, 'abs_z_abs_u': mod_z * mod_u, 'arg_p': arg_p, 'arg_z_plus_arg_u': wrap_deg(arg_z + arg_u)}


def roots_of_unity(n: int) -> list[complex]:
    """The ``n`` solutions of z**n = 1: exp(j 2 pi k / n) for k = 0 .. n - 1."""
    return [cmath.exp(2j * math.pi * k / n) for k in range(n)]


def load_study(v: complex, loads: dict[str, complex]) -> list[dict[str, float | complex | str]]:
    """Current, complex power, |S| two ways and the sign of Q for each load of ``loads`` (Lab step 4)."""
    rows = []
    for name, z in loads.items():
        i = load_current(v, z)
        s = complex_power(v, i)
        q_sign = 'Q > 0 (lagging, absorbs reactive power)' if s.imag > 1e-9 else 'Q = 0'
        if s.imag < -1e-9:
            q_sign = 'Q < 0 (leading, supplies reactive power)'
        rows.append(
            {'load': name, 'z': z, 'i': i, 's': s, 'abs_s': abs(s), 'hypot': math.hypot(s.real, s.imag), 'q': q_sign}
        )
    return rows


def exercise1() -> complex:
    """The quotient (3 + 4j)(1 - j) / (2j) of Exercise 1."""
    return (3 + 4j) * (1 - 1j) / (2j)


def atan_vs_atan2(z: complex) -> tuple[float, float]:
    """``atan(y / x)`` and ``atan2(y, x)`` in degrees for the same point, to see where the first one fails."""
    return math.degrees(math.atan(z.imag / z.real)), math.degrees(math.atan2(z.imag, z.real))


def tap_phase_shift(vn: float, n: int, s: float, phi_deg: float) -> tuple[float, float]:
    """Phase shift in degrees of a tap changer, two ways that must agree (Exercise 4).

    The added voltage is du = vn * n * s / 100 at the step angle phi. The first value is
    arg(1 + (du / vn) exp(j phi)), the second atan2(du sin(phi), vn + du cos(phi)), the formula ToOp uses.
    """
    du = vn * n * s / 100
    phi = math.radians(phi_deg)
    by_arg = math.degrees(cmath.phase(1 + du / vn * cmath.exp(1j * phi)))
    by_atan2 = math.degrees(math.atan2(du * math.sin(phi), vn + du * math.cos(phi)))
    return by_arg, by_atan2


def rotating_phasor(omega: float, t: float) -> tuple[complex, float]:
    """The unit phasor exp(j omega t) and its projection on the real axis, cos(omega t) (Exercise 5)."""
    u = cmath.exp(1j * omega * t)
    return u, u.real


def main() -> None:
    """Print the numbers quoted in the Chapter 1 solutions."""
    z, u = 3 + 4j, 1 + 0.5j
    print('Lab 1: z =', z, 'abs, arg =', polar_deg(z))
    for theta in (0, 30, 90, 180, 270):
        print(f'  theta = {theta:3d}  w = {rotate(z, theta):.4f}  abs, arg = {polar_deg(rotate(z, theta))}')
    print('Lab 2: u =', u, 'abs, arg =', polar_deg(u), ' p = z u =', z * u, product_check(z, u))
    roots = roots_of_unity(3)
    print('Lab 3: cube roots', [f'{r:.4f}' for r in roots], 'sum =', f'{sum(roots):.2e}')
    print('Lab 4:')
    for row in load_study(230, LOADS):
        print(
            f'  {row["load"]:10s} Z = {row["z"]}  I = {row["i"]:.4f}  S = {row["s"]:.1f}  |S| = {row["abs_s"]:.1f}  '
            f'hypot = {row["hypot"]:.1f}  {row["q"]}'
        )
    print('Exercise 1:', exercise1(), 'polar', polar_deg(exercise1()))
    print('Exercise 2: atan(y/x), atan2(y, x) for -1 - j:', atan_vs_atan2(-1 - 1j))
    a = roots[1]
    print('Exercise 3: a =', f'{a:.4f}', ' a^2 =', f'{a**2:.4f}', ' 1 + a + a^2 =', f'{1 + a + a**2:.2e}')
    for phi in (90, 60):
        print(f'Exercise 4: phi = {phi}  shift = {tap_phase_shift(400, 5, 1, phi)}')
    omega = 2 * math.pi
    print(
        'Exercise 5: t, Re u, cos(omega t):',
        [(t, round(rotating_phasor(omega, t)[1], 4), round(math.cos(omega * t), 4)) for t in (0, 0.125, 0.25, 0.5)],
    )


if __name__ == '__main__':
    main()
