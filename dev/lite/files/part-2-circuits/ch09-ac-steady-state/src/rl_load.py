"""A series RL load on a 230 V, 50 Hz supply and the capacitor that corrects its power factor (Chapter 9).

Z = R + jωL, I = V / Z, S = V I* = P + jQ. A capacitor in parallel supplies Qc = Q − P tan(acos pf_target),
so C = Qc / (ω V²).
"""

import cmath
import math


def impedance(r: float, l_henry: float, f_hz: float) -> complex:
    """Series RL impedance R + jωL at frequency ``f_hz``."""
    return complex(r, 2 * math.pi * f_hz * l_henry)


def load_power(v_rms: float, z: complex) -> tuple[complex, complex]:
    """Phasor current and complex power of an impedance fed by ``v_rms`` at angle zero."""
    i = v_rms / z
    return i, v_rms * i.conjugate()


def power_factor(s: complex) -> float:
    """P / |S| of a complex power, positive for the lagging load considered here."""
    return s.real / abs(s)


def correction_capacitor(s: complex, pf_target: float, v_rms: float, f_hz: float) -> tuple[float, float]:
    """Reactive power a parallel capacitor must supply to reach ``pf_target``, and its capacitance in farad."""
    q_target = s.real * math.tan(math.acos(pf_target))
    q_c = s.imag - q_target
    return q_c, q_c / (2 * math.pi * f_hz * v_rms**2)


def main() -> None:
    """Print every number of Exercise 1."""
    v, f = 230.0, 50.0
    z = impedance(10.0, 0.050, f)
    i, s = load_power(v, z)
    q_c, c = correction_capacitor(s, 0.95, v, f)
    print(f'Z = {z.real:.2f} + j{z.imag:.2f} ohm, |Z| = {abs(z):.2f} ohm')
    print(f'|I| = {abs(i):.2f} A at {math.degrees(cmath.phase(i)):.1f} deg')
    print(f'S = {s.real:.0f} + j{s.imag:.0f} VA, |S| = {abs(s):.0f} VA, pf = {power_factor(s):.3f}')
    print(f'Qc = {q_c:.0f} var, C = {1e6 * c:.1f} uF')
    _, s_new = load_power(v, 1 / (1 / z + 1j * 2 * math.pi * f * c))
    print(f'check: with C in parallel, S = {s_new.real:.0f} + j{s_new.imag:.0f} VA, pf = {power_factor(s_new):.3f}')


if __name__ == '__main__':
    main()
