"""Exercise 6 of Chapter 20: the angle differences of the Chapter 19 AC N-1, above 20 and 10 degrees.

Run it with ToOp's interpreter from this folder; it reuses the Chapter 19 module ``ac_n1_toop.py`` (found two
folders up, in ``ch19-operational-security/src``) to run the AC N-1 of ``data/grid_node_breaker/grid.xiidm``, then
lists the ``va_diff_results`` rows above the two thresholds and the worst one. ``above`` and ``worst`` are plain
pandas and need no ToOp; ``main`` does.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

CH19_SRC = Path(__file__).resolve().parents[2] / 'ch19-operational-security' / 'src'
THRESHOLDS = (20.0, 10.0)


def above(va_diff: pd.DataFrame, degrees: float) -> pd.DataFrame:
    """The rows whose |va_diff| exceeds ``degrees``, worst first (NaN rows, an islanded end, are dropped)."""
    rows = va_diff[va_diff.va_diff.abs() > degrees]
    return rows.reindex(rows.va_diff.abs().sort_values(ascending=False).index)


def worst(va_diff: pd.DataFrame) -> pd.Series:
    """The row with the largest |va_diff|."""
    return va_diff.loc[va_diff.va_diff.abs().idxmax()]


def main() -> None:
    """Run the Chapter 19 AC N-1 and report the angle differences."""
    sys.path.insert(0, str(CH19_SRC))
    import ac_n1_toop
    import pypowsybl

    net = pypowsybl.network.load(str(ac_n1_toop.default_grid_path()))
    results = ac_n1_toop.run(net, ac_n1_toop.n1_definition(net))
    va = ac_n1_toop.frames(results)['va_diff']
    print(
        f'{len(va)} va_diff rows (one per outaged branch), {int(va.va_diff.isna().sum())} NaN',
        '(an end left without voltage)',
    )
    for degrees in THRESHOLDS:
        rows = above(va, degrees)
        print(f'rows above {degrees:.0f} deg: {len(rows)}')
    w = worst(va)
    print(f'worst: contingency {w.contingency}, element {w.element}, va_diff {w.va_diff:.3f} deg')
    print(
        va[['contingency', 'element', 'va_diff']]
        .sort_values('va_diff', key=abs, ascending=False)
        .to_string(index=False)
    )


if __name__ == '__main__':
    main()
