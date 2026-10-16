"""Power transfer through a lossless reactance and the error of its DC approximation (Chapter 9).

Two buses, V1 at angle delta and V2 at angle 0, joined by a reactance X. The complex power leaving bus 1 is
S12 = V1 V2 sin(delta) / X + j (V1² − V1 V2 cos(delta)) / X; the DC model keeps P12 ≈ delta / X with flat
voltages and small angles.
"""

import numpy as np
import pandas as pd


def transfer_power(v1: float, v2: float, x: float, delta_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Active and reactive power leaving bus 1 through the reactance ``x`` for an angle difference in degrees."""
    delta = np.radians(delta_deg)
    p = v1 * v2 * np.sin(delta) / x
    q = (v1**2 - v1 * v2 * np.cos(delta)) / x
    return p, q


def phasor_current(v1: float, v2: float, x: float, delta_deg: float) -> complex:
    """Phasor current from bus 1 to bus 2, I = (V1 e^{j delta} − V2) / (jX)."""
    u1 = v1 * np.exp(1j * np.radians(delta_deg))
    return complex((u1 - v2) / (1j * x))


def dc_estimate(x: float, delta_deg: np.ndarray | float) -> np.ndarray:
    """The DC approximation P ≈ delta / X, with delta in radians and flat 1 pu voltages."""
    return np.radians(delta_deg) / x


# --8<-- [start:table]
def error_table(v1: float = 1.0, v2: float = 1.0, x: float = 0.1, step_deg: int = 5) -> pd.DataFrame:
    """Exact P, DC estimate and relative error for delta = 0, step, ..., 90 degrees.

    The relative error splits exactly into two factors: ``angle_error`` = delta / sin(delta) − 1, which is all
    there is when V1 = V2 = 1, and ``magnitude_error`` = 1 / (V1 V2) − 1, from assuming flat voltages, with
    (1 + total) = (1 + angle)(1 + magnitude).
    """
    delta = np.arange(0, 91, step_deg, dtype=float)
    p_exact, _ = transfer_power(v1, v2, x, delta)
    p_dc = dc_estimate(x, delta)
    magnitude = 1 / (v1 * v2) - 1
    with np.errstate(invalid='ignore', divide='ignore'):  # at delta = 0 both powers are 0: take the limit values
        rel = np.where(delta > 0, p_dc / p_exact - 1, magnitude)
        angle = np.where(delta > 0, np.radians(delta) / np.sin(np.radians(delta)) - 1, 0.0)
    return pd.DataFrame(
        {
            'delta_deg': delta,
            'p_exact_pu': p_exact,
            'p_dc_pu': p_dc,
            'rel_error_pct': 100 * rel,
            'angle_error_pct': 100 * angle,
            'magnitude_error_pct': np.full_like(delta, 100 * magnitude),
        }
    )


# --8<-- [end:table]


def dc_limit_overestimate(power_factor: float) -> float:
    """By how much (fraction) P_max = V I_max overstates the admissible active power at a given power factor.

    The admissible active power is V I_max × pf, so the DC limit is 1 / pf − 1 too large relative to it.
    """
    return 1 / power_factor - 1


def main() -> None:
    """Print the numbers quoted in the Chapter 9 worked example, lab and exercises 4 and 6."""
    p, q = transfer_power(1.0, 1.0, 0.1, 30.0)
    print(
        f'worked example: delta = 30 deg, P = {p:.3f} pu, Q = {q:.3f} pu, DC estimate {dc_estimate(0.1, 30.0):.4f} pu'
    )
    print(f'maximum transferable power V1 V2 / X = {transfer_power(1.0, 1.0, 0.1, 90.0)[0]:.1f} pu at delta = 90 deg')
    i = phasor_current(1.0, 1.0, 0.1, 30.0)
    u_diff = np.exp(1j * np.radians(30.0)) - 1.0
    print(f'lab 2: angle(I) - angle(U1 - U2) = {np.degrees(np.angle(i) - np.angle(u_diff)):.1f} deg')
    table = error_table()
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    for d in (10, 30, 60):
        row = table[table.delta_deg == d].iloc[0]
        print(f'lab 3: delta = {d:2d} deg, relative error {row.rel_error_pct:.2f} %')
    table2 = error_table(1.05, 0.95)
    for d in (10, 30, 60):
        row = table2[table2.delta_deg == d].iloc[0]
        print(
            f'lab 4: delta = {d:2d} deg, total {row.rel_error_pct:.2f} %, angle {row.angle_error_pct:.2f} %, '
            f'magnitude {row.magnitude_error_pct:.2f} %'
        )
    for v1, v2 in ((1.05, 1.0), (1.0, 1.05)):
        print(f'exercise 4: delta = 0, V1 = {v1}, V2 = {v2}: Q12 = {transfer_power(v1, v2, 0.1, 0.0)[1]:+.3f} pu')
    print(f'exercise 4: Q12 with flat voltages at 10 deg = {transfer_power(1, 1, 0.1, 10.0)[1]:.4f} pu, second order')
    print(f'exercise 6: DC limit overestimate at pf 0.95 = {100 * dc_limit_overestimate(0.95):.2f} %')


if __name__ == '__main__':
    main()
