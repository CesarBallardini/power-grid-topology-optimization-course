"""The same query on pandapower's case57 line results in pandas and in Polars (Chapter 30, Exercise 4).

Keep the lines whose loading is above the median, then average the loading per from bus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandapower as pp
import pandapower.networks as pn

if TYPE_CHECKING:
    import pandas as pd


def res_line_case57() -> pd.DataFrame:
    """``net.res_line`` of case57 after an AC power flow, with the ``from_bus`` of every line joined in."""
    net = pn.case57()
    pp.runpp(net)
    return net.res_line.join(net.line[['from_bus', 'to_bus']])


def pandas_query(df: pd.DataFrame) -> pd.Series:
    """Mean loading per from bus of the lines above the median loading, eager pandas style."""
    above = df[df['loading_percent'] > df['loading_percent'].median()]
    return above.groupby('from_bus')['loading_percent'].mean()


def polars_query(df: pd.DataFrame):
    """The same result with a Polars LazyFrame: nothing runs until ``collect()``."""
    import polars as pl

    query = (
        pl.from_pandas(df.reset_index(names='line'))
        .lazy()
        .filter(pl.col('loading_percent') > pl.col('loading_percent').median())
        .group_by('from_bus')
        .agg(pl.col('loading_percent').mean())
        .sort('from_bus')
    )
    return query.collect()


def main() -> None:
    """Print the median, the pandas result and, when Polars is installed, the Polars result."""
    df = res_line_case57()
    print(f'{len(df)} lines, median loading {df["loading_percent"].median():.2f} %')
    result = pandas_query(df)
    print(f'{len(result)} from buses with at least one line above the median; the five most loaded:')
    print(result.sort_values(ascending=False).head().round(2).to_string())
    try:
        print(polars_query(df).head())
    except ImportError:
        print("Polars is not installed in this environment; run the script with ToOp's interpreter to see it")


if __name__ == '__main__':
    main()
