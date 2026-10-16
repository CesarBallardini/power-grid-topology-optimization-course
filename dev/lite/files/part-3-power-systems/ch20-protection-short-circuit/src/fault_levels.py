"""Fault levels by hand (Chapter 20): the IEC 60909 equivalent voltage source, Z_bus and a busbar split.

The worked example of the chapter (two infeeds on a double busbar, coupler closed or open), the single
line-to-ground current of Exercise 1 and the five-bus Z_bus calculation of Exercise 2, with a split node. All
impedances are in ohm at the fault voltage level, currents in kA, and the voltage factor c multiplies the
nominal line-to-line voltage: I''k = c Un / (sqrt(3) |Zk|).
"""

from __future__ import annotations  # ToOp's interpreter is older than 3.14: ``split`` returns its own class

from dataclasses import dataclass

import numpy as np


def feeder_impedance(skq_mva: float, un_kv: float = 220.0, c: float = 1.1) -> float:
    """Network feeder impedance Z_Q = c Un^2 / S''kQ in ohm."""
    return c * un_kv**2 / skq_mva


def ikss_ka(zk_ohm: float, un_kv: float = 220.0, c: float = 1.1) -> float:
    """Initial symmetrical short-circuit current c Un / (sqrt(3) Zk) in kA."""
    return c * un_kv / (np.sqrt(3) * zk_ohm)


def skss_mva(ik_ka: float, un_kv: float = 220.0) -> float:
    """Short-circuit power sqrt(3) Un I''k in MVA."""
    return np.sqrt(3) * un_kv * ik_ka


def worked_example(skq=(8000.0, 6000.0), km=(40.0, 60.0), x_per_km=0.4, un_kv=220.0, c=1.1) -> dict[str, float]:
    """The chapter's station: path impedances, the parallel fault level and the two split fault levels."""
    zq = [feeder_impedance(s, un_kv, c) for s in skq]
    z = [q + x_per_km * d for q, d in zip(zq, km, strict=True)]
    zk = 1.0 / sum(1.0 / zi for zi in z)
    return {
        'zq1_ohm': zq[0],
        'zq2_ohm': zq[1],
        'z1_ohm': z[0],
        'z2_ohm': z[1],
        'zk_closed_ohm': zk,
        'ik_closed_ka': ikss_ka(zk, un_kv, c),
        'ik_open_busbar1_ka': ikss_ka(z[0], un_kv, c),
        'ik_open_busbar2_ka': ikss_ka(z[1], un_kv, c),
    }


def three_phase_current(e: complex, z1: complex) -> complex:
    """Three-phase fault current E / Z1 (positive sequence only)."""
    return e / z1


def slg_current(e: complex, z1: complex, z2: complex, z0: complex) -> complex:
    """Single line-to-ground fault current 3 E / (Z1 + Z2 + Z0): the three sequence networks in series."""
    return 3 * e / (z1 + z2 + z0)


@dataclass(frozen=True)
class Grid:
    """A small grid for Z_bus: branches (from, to, x ohm) and source impedances to ground per bus (ohm, 0 = none)."""

    branches: list[tuple[int, int, float]]
    sources: dict[int, float]
    n_bus: int

    def ybus(self) -> np.ndarray:
        """Nodal admittance matrix in siemens (reactances only, so it is real and the j is left implicit)."""
        y = np.zeros((self.n_bus, self.n_bus))
        for i, j, x in self.branches:
            y[i, i] += 1 / x
            y[j, j] += 1 / x
            y[i, j] -= 1 / x
            y[j, i] -= 1 / x
        for i, x in self.sources.items():
            y[i, i] += 1 / x
        return y

    def zbus(self) -> np.ndarray:
        """Z_bus = Y_bus^-1; its diagonal is the Thevenin impedance at each bus."""
        return np.linalg.inv(self.ybus())

    def fault_currents(self, un_kv: float, c: float = 1.1) -> np.ndarray:
        """I''k at every bus in kA, from the diagonal of Z_bus."""
        return np.array([ikss_ka(z, un_kv, c) for z in np.diag(self.zbus())])

    def split(self, bus: int, moved: list[int]) -> Grid:
        """Split ``bus``: branches with the given indices move their ``bus`` end to a new node, as in Chapter 18."""
        new = self.n_bus
        branches = []
        for k, (i, j, x) in enumerate(self.branches):
            if k in moved:
                i, j = (new if i == bus else i), (new if j == bus else j)
            branches.append((i, j, x))
        return Grid(branches=branches, sources=dict(self.sources), n_bus=self.n_bus + 1)


# Exercise 2: five 110 kV buses, two infeeds (buses 0 and 4), a meshed middle; reactances in ohm.
FIVE_BUS = Grid(
    branches=[(0, 1, 8.0), (1, 2, 6.0), (2, 3, 6.0), (3, 4, 8.0), (1, 3, 10.0), (0, 2, 12.0)],
    sources={0: 2.0, 4: 3.0},
    n_bus=5,
)


def split_comparison(grid: Grid = FIVE_BUS, bus: int = 2, moved: list[int] | None = None, un_kv: float = 110.0):
    """Exercise 2: fault currents before and after moving two branches of ``bus`` to a new node, and the changes."""
    moved = [2, 5] if moved is None else moved  # branches 2-3 and 0-2 leave bus 2 for the new node
    before = grid.fault_currents(un_kv)
    after = grid.split(bus, moved).fault_currents(un_kv)
    change = 100 * (after[: grid.n_bus] - before) / before
    return before, after, change


def main() -> None:
    """Print the worked example, the SLG check and the five-bus split."""
    for key, value in worked_example().items():
        print(f'{key:22s} {value:8.3f}')
    e, z1 = 1.0, 0.2j
    for z0 in (0.1j, 0.2j, 0.6j):
        ratio = abs(slg_current(e, z1, z1, z0)) / abs(three_phase_current(e, z1))
        print(f'Z0 = {z0.imag:.1f} Z1 (Z2 = Z1): I_SLG / I_3ph = {ratio:.3f}')
    before, after, change = split_comparison()
    print("five-bus I''k before the split (kA):", np.round(before, 3))
    print('after splitting bus 3 (0-based 2), new node last (kA):', np.round(after, 3))
    print(
        'change in % per original bus:',
        np.round(change, 1),
        '; buses changed by more than 10 %:',
        np.flatnonzero(np.abs(change) > 10).tolist(),
    )


if __name__ == '__main__':
    main()
