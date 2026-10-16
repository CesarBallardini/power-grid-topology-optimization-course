"""Impedances across an ideal transformer, ToOp's x/ρ rule and the star equivalent of a three-winding unit.

An impedance Z on the side at voltage V2 is seen from the side at V1 as Z (V1/V2)². ToOp divides a transformer's
DC reactance by its ratio ρ, so the branch susceptance becomes ρ/x: a ratio above 1 raises it by ρ − 1. The
pairwise short-circuit reactances x12, x13, x23 of a three-winding transformer give the star branches
x1 = (x12 + x13 − x23)/2 and its cyclic permutations.
"""

import numpy as np
from dc_nodal import nodal_solve


def referred_impedance(z: complex, v_from_kv: float, v_to_kv: float) -> complex:
    """Impedance ``z`` on the ``v_to_kv`` side as seen from the ``v_from_kv`` side: z (V_from / V_to)²."""
    return z * (v_from_kv / v_to_kv) ** 2


def susceptance_change(rho: float) -> float:
    """Relative change of the DC susceptance when x is replaced by x/ρ: (ρ/x) / (1/x) − 1 = ρ − 1."""
    return rho - 1


def flow_share_with_ratio(rho: float, x_t: float = 0.1, x_line: float = 0.3) -> float:
    """Share of a 1 pu transfer carried by a transformer of reactance ``x_t`` / ρ in parallel with a line."""
    _, flows = nodal_solve([0, 0], [1, 1], [rho / x_t, 1 / x_line], [1.0, -1.0], 2)
    return float(flows[0])


def star_equivalent(x12: float, x13: float, x23: float) -> tuple[float, float, float]:
    """Star reactances (x1, x2, x3) from the pairwise short-circuit reactances."""
    return (x12 + x13 - x23) / 2, (x12 + x23 - x13) / 2, (x13 + x23 - x12) / 2


def main() -> None:
    """Print the numbers of Exercises 1, 2 and 3."""
    print(f'exercise 1: 10 ohm at 110 kV seen from 380 kV = {referred_impedance(10.0, 380.0, 110.0):.1f} ohm')
    print(
        f'exercise 2: rho = 1.05 changes the susceptance by {100 * susceptance_change(1.05):+.1f} %; '
        f'flow share {flow_share_with_ratio(1.0):.4f} -> {flow_share_with_ratio(1.05):.4f} of 1 pu'
    )
    x1, x2, x3 = star_equivalent(0.10, 0.25, 0.20)
    print(
        f'exercise 3: x1 = {x1:.3f}, x2 = {x2:.3f}, x3 = {x3:.3f} pu; check x1 + x2 = {x1 + x2:.2f}, '
        f'x1 + x3 = {x1 + x3:.2f}, x2 + x3 = {x2 + x3:.2f}'
    )
    print(np.round([x1, x2, x3], 3))


if __name__ == '__main__':
    main()
