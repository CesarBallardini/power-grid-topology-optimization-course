"""Balanced three-phase phasors, wye loads, instantaneous power, relay impedance and symmetrical components.

The operator a = e^{j2π/3} rotates a phasor by 120°. A balanced set is (V, a² V, a V); the line voltage
V_ab = V_a − V_b has magnitude √3 |V| and leads V_a by 30°. A wye load Z_Y per phase draws I = V_ph / Z_Y per
line and P + jQ = 3 |I|² Z_Y in total; the equivalent delta is Z_Δ = 3 Z_Y. The relay model of Chapter 10,
Exercise 3 takes |Z| = V_ph / I and the angle atan2(Q, P). The symmetrical components are V_012 = A⁻¹ V_abc.
"""

import cmath
import math

import numpy as np

A_OP = cmath.exp(2j * math.pi / 3)  # the 120-degree rotation
A_MATRIX = np.array([[1, 1, 1], [1, A_OP**2, A_OP], [1, A_OP, A_OP**2]])  # V_abc = A V_012


def balanced_set(magnitude: float = 1.0, angle_deg: float = 0.0) -> np.ndarray:
    """Phasors (V_a, V_b, V_c) of a balanced positive-sequence set, b lagging a by 120 degrees."""
    v_a = magnitude * cmath.exp(1j * math.radians(angle_deg))
    return np.array([v_a, v_a * A_OP**2, v_a * A_OP])


def line_voltages(v_abc: np.ndarray) -> np.ndarray:
    """Line-to-line phasors (V_ab, V_bc, V_ca) from the phase phasors."""
    return v_abc - np.roll(v_abc, -1)


def wye_load(z_y: complex, v_ll: float) -> tuple[complex, complex, complex]:
    """Line current (phase a), total complex power and the equivalent delta impedance of a balanced wye load."""
    v_ph = v_ll / math.sqrt(3)
    i_a = v_ph / z_y
    return i_a, 3 * abs(i_a) ** 2 * z_y, 3 * z_y


def instantaneous_power(v_rms: float, i_rms: float, phi_deg: float, f_hz: float = 50.0, n: int = 2000) -> tuple:
    """Time axis, the three phase powers p_a, p_b, p_c and their sum over one period, for a balanced load."""
    t = np.linspace(0.0, 1 / f_hz, n, endpoint=False)
    w, phi = 2 * np.pi * f_hz, np.radians(phi_deg)
    shifts = np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3])[:, None]
    v = np.sqrt(2) * v_rms * np.cos(w * t + shifts)
    i = np.sqrt(2) * i_rms * np.cos(w * t + shifts - phi)
    p = v * i
    return t, p, p.sum(axis=0)


def relay_impedance(v_ll_kv: float, i_a: float, p_mw: float, q_mvar: float) -> tuple[float, float, float]:
    """|Z|, R and X seen by a relay: phase voltage V_LL / √3 over the current, at the angle atan2(Q, P)."""
    z_abs = 1e3 * v_ll_kv / math.sqrt(3) / i_a
    angle = math.atan2(q_mvar, p_mw)
    return z_abs, z_abs * math.cos(angle), z_abs * math.sin(angle)


def symmetrical_components(v_abc: np.ndarray) -> np.ndarray:
    """Zero, positive and negative sequence components V_012 = A⁻¹ V_abc."""
    return np.linalg.solve(A_MATRIX, v_abc)


def main() -> None:
    """Print the numbers of the lab steps 1 and 2 and of Exercises 1, 2, 3 and 5."""
    v = balanced_set()
    v_ab = line_voltages(v)[0]
    deg = math.degrees(cmath.phase(v_ab))
    print(f'lab 1: |V_ab| = {abs(v_ab):.4f} = sqrt(3) = {math.sqrt(3):.4f}, arg(V_ab) = {deg:.1f} deg')
    i = balanced_set(1.0, -30.0)
    print(f'lab 2: I_a + I_b + I_c = {abs(i.sum()):.1e} for phi = 30 deg')
    i_a, s, z_delta = wye_load(30 + 40j, 400.0)
    deg = math.degrees(cmath.phase(i_a))
    print(f'exercise 1: V_ph = {400 / math.sqrt(3):.2f} V, |I_L| = {abs(i_a):.3f} A at {deg:.2f} deg')
    print(f'exercise 1: P = {s.real:.0f} W, Q = {s.imag:.0f} var, |S| = {abs(s):.0f} VA, Z_delta = {z_delta}')
    _, p_phase, p_total = instantaneous_power(230.0, 10.0, 30.0)
    swing, constant = (p_phase[0].max() - p_phase[0].min()) / 2, 3 * 2300 * math.cos(math.radians(30))
    print(
        f'exercise 2: per-phase swing {swing:.1f} W, total min {p_total.min():.1f} W, max {p_total.max():.1f} W, '
        f'3 V I cos(phi) = {constant:.1f} W'
    )
    z_abs, r, x = relay_impedance(380.0, 1000.0, 600.0, 200.0)
    angle = math.degrees(math.atan2(200, 600))
    print(f'exercise 3: |Z| = {z_abs:.1f} ohm, angle = {angle:.2f} deg, R = {r:.1f} ohm, X = {x:.1f} ohm')
    v_012 = symmetrical_components(np.array([1.0, 0.9 * A_OP**2, A_OP]))
    print(f'exercise 5: |V_0| = {abs(v_012[0]):.4f}, |V_1| = {abs(v_012[1]):.4f}, |V_2| = {abs(v_012[2]):.4f} pu')


if __name__ == '__main__':
    main()
