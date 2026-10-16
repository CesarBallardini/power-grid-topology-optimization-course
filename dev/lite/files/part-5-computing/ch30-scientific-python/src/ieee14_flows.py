"""AC and DC active line flows of pypowsybl's IEEE 14 case, sorted by magnitude (Chapter 30, Exercise 3)."""

import matplotlib.pyplot as plt
import pandas as pd
import pypowsybl as pypo


def _converged(results) -> None:
    """Stop if the load flow of the main synchronous component did not converge."""
    if results[0].status != pypo.loadflow.ComponentStatus.CONVERGED:
        raise RuntimeError(f'load flow did not converge: {results[0].status}')


def line_flows() -> pd.DataFrame:
    """Active power at the from end of every line after an AC and after a DC load flow.

    Returns:
        Frame indexed by line id with the columns ``p1_ac`` and ``p1_dc`` (MW), sorted by decreasing |p1_ac|.
        The DC run overwrites the same ``p1`` column of ``get_lines()``, which is why the AC values are copied
        first.
    """
    n = pypo.network.create_ieee14()
    _converged(pypo.loadflow.run_ac(n))
    p1_ac = n.get_lines()['p1'].copy()
    _converged(pypo.loadflow.run_dc(n))
    p1_dc = n.get_lines()['p1']
    df = pd.DataFrame({'p1_ac': p1_ac, 'p1_dc': p1_dc})
    return df.reindex(df['p1_ac'].abs().sort_values(ascending=False).index)


def bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Two bar series side by side, one line per group of bars, in the order of the frame."""
    fig, ax = plt.subplots(figsize=(9, 4))
    pos = range(len(df))
    ax.bar([p - 0.2 for p in pos], df['p1_ac'], width=0.4, label='AC')
    ax.bar([p + 0.2 for p in pos], df['p1_dc'], width=0.4, label='DC')
    ax.set_xticks(list(pos), df.index, rotation=90)
    ax.set_ylabel('p1 [MW]')
    ax.set_title('IEEE 14: active power at the from end of every line')
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    """Print the sorted table and the largest AC-DC gap."""
    df = line_flows()
    print(df.round(2).to_string())
    gap = (df['p1_ac'] - df['p1_dc']).abs()
    print(f'largest |AC - DC| gap: {gap.max():.2f} MW on {gap.idxmax()}; mean gap {gap.mean():.2f} MW')


if __name__ == '__main__':
    main()
