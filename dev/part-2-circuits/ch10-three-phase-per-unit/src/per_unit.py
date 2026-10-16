"""Per-unit bases, change of base and the current-to-power conversion of branch limits (Chapter 10).

With a three-phase base power S_base (MVA) and a line-to-line base voltage V_base (kV): Z_base = V_base² / S_base
in ohm, I_base = S_base / (√3 V_base) in kA, z_pu = Z / Z_base, and a per-unit value moves to a new base as
z_new = z_old (V_old / V_new)² (S_new / S_old). A current limit I (A) at a voltage V (kV) is a power limit
√3 V I × 1e-3 MW at unity power factor, the formula of ToOp's powsybl_helpers.py.
"""

import math

import pandas as pd

S_BASE_MVA = 100.0


def z_base(v_kv: float, s_mva: float = S_BASE_MVA) -> float:
    """Base impedance V² / S in ohm, V line-to-line in kV and S three-phase in MVA."""
    return v_kv**2 / s_mva


def i_base(v_kv: float, s_mva: float = S_BASE_MVA) -> float:
    """Base current S / (√3 V) in ampere."""
    return 1e3 * s_mva / (math.sqrt(3) * v_kv)


def change_base(z_pu: float, v_old_kv: float, s_old_mva: float, v_new_kv: float, s_new_mva: float) -> float:
    """A per-unit impedance moved from one (voltage, power) base to another."""
    return z_pu * (v_old_kv / v_new_kv) ** 2 * (s_new_mva / s_old_mva)


def p_limit_mw(i_max_a: float, v_kv: float, power_factor: float = 1.0) -> float:
    """Active power a three-phase branch carries at its current limit: √3 V I × 1e-3 × pf, V in kV and I in A."""
    return i_max_a * v_kv * 1e-3 * math.sqrt(3) * power_factor


# --8<-- [start:table]
BRANCHES = pd.DataFrame(
    {
        'name': ['380 kV line', '220 kV line', '110 kV cable'],
        'v_kv': [380.0, 220.0, 110.0],
        'length_km': [150.0, 100.0, 20.0],
        'r_ohm_per_km': [0.03, 0.06, 0.05],
        'x_ohm_per_km': [0.30, 0.40, 0.12],
        'i_max_a': [2000.0, 1000.0, 600.0],
    }
)


def branch_table(branches: pd.DataFrame = BRANCHES, s_mva: float = S_BASE_MVA) -> pd.DataFrame:
    """Add Z_base, the per-unit resistance and reactance on ``s_mva`` and the MW limit at unity power factor."""
    out = branches.copy()
    out['z_base_ohm'] = out.v_kv**2 / s_mva
    out['r_pu'] = out.r_ohm_per_km * out.length_km / out.z_base_ohm
    out['x_pu'] = out.x_ohm_per_km * out.length_km / out.z_base_ohm
    out['p_limit_mw'] = out.i_max_a * out.v_kv * 1e-3 * math.sqrt(3)
    return out


# --8<-- [end:table]


def main() -> None:
    """Print the worked example, the lab table and the numbers of Exercise 4."""
    print(f'380 kV, 1000 A: P = {p_limit_mw(1000.0, 380.0):.1f} MW at unity power factor')
    print(
        f'220 kV on 100 MVA: Z_base = {z_base(220.0):.0f} ohm, x_pu of 40 ohm = {40 / z_base(220.0):.4f}, '
        f'I_base = {i_base(220.0):.1f} A'
    )
    print(f'transformer 0.10 pu on 50 MVA -> {change_base(0.10, 1.0, 50.0, 1.0, 100.0):.2f} pu on 100 MVA')
    table = branch_table()
    print(table.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    x_new = change_base(0.15, 1.0, 1600.0, 1.0, 100.0)
    print(f'exercise 4: x = 0.15 pu on 1600 MVA -> {x_new:.6f} pu on 100 MVA ({x_new * 100:.4f} %)')
    z_hv, z_lv = z_base(380.0, 1600.0), z_base(110.0, 1600.0)
    print(
        f'exercise 4: Z_base at 380 kV = {z_hv:.3f} ohm, at 110 kV = {z_lv:.4f} ohm, ratio {z_hv / z_lv:.3f} '
        f'= (380/110)^2 = {(380 / 110) ** 2:.3f}'
    )


if __name__ == '__main__':
    main()
