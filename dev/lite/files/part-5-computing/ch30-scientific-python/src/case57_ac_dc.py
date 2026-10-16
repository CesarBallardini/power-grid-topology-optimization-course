"""AC and DC from-end flows of pandapower's case57 side by side, in pandas and Polars, with a histogram (Lab 4)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from two_libraries import pandapower_case57


def side_by_side() -> pd.DataFrame:
    """One row per line of case57: AC flow, DC flow and their difference, all in MW at the from end."""
    _, ac_p, dc_p = pandapower_case57()
    df = pd.DataFrame({'ac_p_from_mw': ac_p, 'dc_p_from_mw': dc_p})
    df['difference_mw'] = df['ac_p_from_mw'] - df['dc_p_from_mw']
    return df


def to_polars(df: pd.DataFrame):
    """The same table as a Polars frame (Polars is in ToOp's environment, not in the course one)."""
    import polars as pl

    return pl.from_pandas(df.reset_index(names='line'))


def histogram(df: pd.DataFrame, bins: int = 20) -> plt.Figure:
    """Histogram of the AC - DC difference of the from-end flows."""
    fig, ax = plt.subplots()
    ax.hist(df['difference_mw'], bins=bins)
    ax.set_xlabel('AC - DC from-end flow [MW]')
    ax.set_ylabel('lines')
    ax.set_title('case57: where the DC approximation is off')
    return fig


def main() -> None:
    """Print the summary of the difference and the histogram counts that the figure shows."""
    df = side_by_side()
    print(df.round(2).head())
    print(df['difference_mw'].describe().round(2))
    worst = df['difference_mw'].abs().idxmax()
    print(f'largest gap: line {worst}, AC {df.ac_p_from_mw[worst]:.1f} MW, DC {df.dc_p_from_mw[worst]:.1f} MW')
    counts, edges = np.histogram(df['difference_mw'], bins=10)
    print('histogram counts:', counts.tolist())
    print('bin edges [MW]:', np.round(edges, 1).tolist())
    try:
        print(to_polars(df).head(3))
    except ImportError:
        print("Polars is not installed in this environment; run the script with ToOp's interpreter to see it")


if __name__ == '__main__':
    main()
