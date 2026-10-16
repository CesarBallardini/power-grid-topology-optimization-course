"""Average cost of German congestion management per MWh from SMARD's quarterly figures (Chapter 12, Lab step 1).

The figures are those of the SMARD article "Congestion management in Q1 2026: lower volume of measures and costs"
(read on 16 October 2026): total volume of measures (redispatch with operational and reserve plants, and
countertrading), estimated total cost, and the part of the redispatch volume that curtailed renewable plants. The
cost breakdown of the same article is kept in a second table; its rows add up to the totals.
"""

import pandas as pd

QUARTERS = pd.DataFrame(
    {
        'quarter': ['Q1 2025', 'Q1 2026'],
        'volume_gwh': [9035.0, 8248.0],
        'cost_meur': [882.0, 784.0],
        'curtailment_gwh': [2153.0, 2215.0],
    }
).set_index('quarter')

COST_BREAKDOWN_MEUR = pd.DataFrame(
    {
        'Q1 2025': [423.0, 71.0, 355.0, 33.0],
        'Q1 2026': [282.0, 55.0, 430.0, 17.0],
    },
    index=[
        'redispatch, conventional plants',
        'compensation, curtailed renewables',
        'reserve plants',
        'countertrading',
    ],
)


def cost_per_mwh(table: pd.DataFrame) -> pd.Series:
    """Average cost of congestion management in euros per MWh of measures: million euros over GWh, times 1000."""
    return 1000.0 * table['cost_meur'] / table['volume_gwh']


def changes(table: pd.DataFrame) -> pd.DataFrame:
    """Absolute and relative change of each column, unit cost included, between the first and the last quarter."""
    with_unit = table.assign(eur_per_mwh=cost_per_mwh(table))
    first, last = with_unit.iloc[0], with_unit.iloc[-1]
    return pd.DataFrame({'first': first, 'last': last, 'change': last - first, 'change_pct': 100 * (last / first - 1)})


def main() -> None:
    """Print the unit cost of each quarter, the changes between them and the cost breakdown."""
    pd.set_option('display.width', 119)
    print(QUARTERS.assign(eur_per_mwh=cost_per_mwh(QUARTERS)).round(2))
    print()
    print(changes(QUARTERS).round(2))
    print()
    print(
        COST_BREAKDOWN_MEUR.assign(
            change_pct=(100 * (COST_BREAKDOWN_MEUR['Q1 2026'] / COST_BREAKDOWN_MEUR['Q1 2025'] - 1))
        ).round(1)
    )
    print('breakdown totals:', COST_BREAKDOWN_MEUR.sum().to_dict())


if __name__ == '__main__':
    main()
