"""Three GLSKs for zone A of the worked example: what moves and what does not (Chapter 21, Exercise 3)."""

import numpy as np
import pandas as pd
from flow_based import BRANCHES, F_MAX, FRM, GLSK, P_REF, flow_based_domain, nodal_ptdf

ZONE_A_KEYS = {'(0.6, 0.4)': (0.6, 0.4), '(1.0, 0.0)': (1.0, 0.0), '(0.0, 1.0)': (0.0, 1.0)}


def glsk_with_zone_a(keys: tuple[float, float]) -> np.ndarray:
    """The GLSK of the worked example with the two shift keys of zone A replaced."""
    g = GLSK.copy()
    g[:2, 0] = np.asarray(keys)
    return g


def compare() -> pd.DataFrame:
    """Per GLSK and branch: zone-to-zone PTDF, F0 and RAM; the reference flow is the same column for all three."""
    ptdf = nodal_ptdf(BRANCHES, 4)
    f_ref = ptdf @ P_REF
    rows = []
    for label, keys in ZONE_A_KEYS.items():
        ptdf_ab, ram, _ = flow_based_domain(ptdf, glsk_with_zone_a(keys), P_REF, F_MAX, FRM)
        f_0 = F_MAX - FRM - ram
        for k, (f, t) in enumerate(BRANCHES):
            rows.append(
                {
                    'glsk': label,
                    'branch': f'{f}-{t}',
                    'f_ref': f_ref[k],
                    'ptdf_ab': ptdf_ab[k],
                    'f_0': f_0[k],
                    'ram': ram[k],
                }
            )
    return pd.DataFrame(rows).set_index(['glsk', 'branch'])


def main() -> None:
    """Print the comparison and the largest A -> B exchange under each GLSK."""
    table = compare()
    print(table.round(3).to_string())
    for label in ZONE_A_KEYS:
        sub = table.loc[label]
        loaded = sub.ptdf_ab > 0  # a branch relieved by the exchange never limits it
        allowed = sub.ram[loaded] / sub.ptdf_ab[loaded]
        print(f'GLSK {label}: largest A->B exchange {allowed.min():.0f} MW on branch {allowed.idxmin()}')


if __name__ == '__main__':
    main()
