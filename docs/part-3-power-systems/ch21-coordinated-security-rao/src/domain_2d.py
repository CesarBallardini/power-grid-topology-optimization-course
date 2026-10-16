"""Five CNECs, three zones, one plane: the flow-based domain of Chapter 21, Exercise 2, in numbers.

GeoGebra draws the half-planes; this module checks the picture. The zonal PTDFs are given relative to the balancing
zone C (its column is zero, zone-to-zone PTDFs are differences), so a CNEC reads ``a NP_A + b NP_B <= RAM``.
"""

import numpy as np
import pandas as pd
from ieee14_flow_based import domain_vertices

CNECS = pd.DataFrame(
    {
        'ptdf_A': [0.5, -0.3, -0.4, 0.2, 0.6],
        'ptdf_B': [0.1, 0.4, -0.5, -0.6, -0.4],
        'ram': [200.0, 180.0, 250.0, 220.0, 150.0],
    },
    index=['CNEC 1', 'CNEC 2', 'CNEC 3', 'CNEC 4', 'CNEC 5'],
)


def exchange_a_to_b(cnecs: pd.DataFrame) -> tuple[float, str]:
    """Largest exchange e with NP = (e, -e, 0): the smallest RAM / (ptdf_A - ptdf_B) over CNECs loaded by it."""
    s = cnecs.ptdf_A - cnecs.ptdf_B
    loaded = s > 1e-12
    ratio = cnecs.ram[loaded] / s[loaded]
    return float(ratio.min()), str(ratio.idxmin())


def with_ram(cnecs: pd.DataFrame, cnec: str, ram: float) -> pd.DataFrame:
    """A copy of the table with one RAM changed: the slider of the exercise."""
    out = cnecs.copy()
    out.loc[cnec, 'ram'] = ram
    return out


def vertices(cnecs: pd.DataFrame) -> np.ndarray:
    """Corners of the shaded domain in the (NP_A, NP_B) plane."""
    return domain_vertices(cnecs[['ptdf_A', 'ptdf_B']].to_numpy(), cnecs.ram.to_numpy())


def main() -> None:
    """The domain before and after the slider raises the RAM of the binding CNEC."""
    e, cnec = exchange_a_to_b(CNECS)
    print(f'vertices: {np.round(vertices(CNECS), 1).tolist()}')
    print(f'largest A->B exchange {e:.0f} MW, bound by {cnec}')
    for ram in (200.0, 300.0, 400.0):
        e2, cnec2 = exchange_a_to_b(with_ram(CNECS, cnec, ram))
        print(f'RAM of {cnec} = {ram:.0f} MW: largest A->B exchange {e2:.0f} MW, bound by {cnec2}')


if __name__ == '__main__':
    main()
