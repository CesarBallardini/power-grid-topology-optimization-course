"""One load on a 230 V source: current and complex power as complex numbers (Chapter 1)."""

import cmath
import math


def load_current(v: complex, z: complex) -> complex:
    """Phasor current drawn by the impedance ``z`` from the voltage phasor ``v``: I = V / Z."""
    return v / z


def complex_power(v: complex, i: complex) -> complex:
    """Complex power S = V I*, whose real part is P (W) and imaginary part Q (var)."""
    return v * i.conjugate()


def main() -> None:
    """Print magnitude and angle of the current and three ways of reading the apparent power."""
    v, z = 230, 40 + 30j
    i = load_current(v, z)
    s = complex_power(v, i)
    print(abs(i), math.degrees(cmath.phase(i)))  # 4.6  -36.87
    print(s, abs(s), math.hypot(s.real, s.imag))  # (846.4+634.8j)  1058.0  1058.0


if __name__ == '__main__':
    main()
