"""Tap tables of a phase-shifting transformer and the linearity test of ToOp's PowSyBl importer (Chapter 11).

Each tap n adds a series voltage of n × s (relative) at the step angle φ, so the shift is
α(n) = atan2(n s sin φ, 1 + n s cos φ). A PST is linear in DC when its reactance is the same at every tap
(``np.allclose`` on the x column, the test of ``get_linear_pst``); a reactance that grows with |n| is not.
"""

import numpy as np
import pandas as pd

TAPS = np.arange(-16, 17)
STEP = 0.01  # 1 % of the nominal voltage per tap
X_NEUTRAL = 0.1  # pu


def shift_deg(taps: np.ndarray, step: float = STEP, phi_deg: float = 90.0) -> np.ndarray:
    """Phase shift per tap from the atan2 formula, in degrees."""
    phi = np.radians(phi_deg)
    return np.degrees(np.arctan2(taps * step * np.sin(phi), 1 + taps * step * np.cos(phi)))


def constant_x_table(taps: np.ndarray = TAPS, x: float = X_NEUTRAL, phi_deg: float = 90.0) -> pd.DataFrame:
    """Tap table with the same reactance at every tap and the atan2 shift."""
    return pd.DataFrame({'tap': taps, 'alpha_deg': shift_deg(taps, phi_deg=phi_deg), 'x_pu': np.full(len(taps), x)})


def growing_x_table(taps: np.ndarray = TAPS, x: float = X_NEUTRAL, growth: float = 0.01) -> pd.DataFrame:
    """Tap table whose reactance grows by ``growth`` (1 %) per tap away from neutral."""
    table = constant_x_table(taps, x)
    table['x_pu'] = x * (1 + growth * np.abs(taps))
    return table


def is_linear_pst(table: pd.DataFrame) -> bool:
    """ToOp's DC test: a PST is linear when every tap has the same reactance."""
    return bool(np.allclose(table.x_pu, table.x_pu.iloc[0]))


def main() -> None:
    """Print the two tables' linearity verdicts and how far α departs from proportionality (Exercise 4)."""
    constant, growing = constant_x_table(), growing_x_table()
    print(f'constant x: linear = {is_linear_pst(constant)}; growing x: linear = {is_linear_pst(growing)}')
    per_tap = constant.alpha_deg / constant.tap.replace(0, np.nan)
    tap1 = per_tap[constant.tap == 1].iloc[0]
    print(f'alpha per tap: {per_tap.min():.4f} to {per_tap.max():.4f} deg (tap 1: {tap1:.4f})')
    a16 = constant.alpha_deg[constant.tap == 16].iloc[0]
    print(f'tap 16: alpha = {a16:.3f} deg against 16 x {tap1:.4f} = {16 * tap1:.3f} deg if proportional')
    print(f'x at tap 16 of the growing table: {growing.x_pu.iloc[-1]:.3f} pu')
    print(constant_x_table(np.array([-16, -8, 0, 8, 16])).to_string(index=False))


if __name__ == '__main__':
    main()
