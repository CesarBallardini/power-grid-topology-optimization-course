"""ToOp's dominator filter next to a true Pareto filter, with the counterexamples of Lab 2 (Chapter 27).

``dominator_mask`` is a line-by-line port of ``get_dominator_mask`` in ToOp's ``optimizer/ac/select_strategy.py``
(commit ba1874e); ``test_dominator_filter_toop.py`` checks it against the original. Fitness is maximized (ToOp's
fitness is a negated cost), every metric is minimized. Run it to print the masks quoted in the solutions::

    python dominator_filter.py
"""

import numpy as np
import pandas as pd


# --8<-- [start:filters]
def dominator_mask(df: pd.DataFrame, target_metrics: list[str], observed_metrics: list[str]) -> np.ndarray:
    """Port of ToOp's ``get_dominator_mask``: True keeps the row.

    For each target metric, in order, and each of its discrete values: take the row with the best fitness among
    the rows of that value that are still kept, and remove every row of that value whose fitness is lower *and*
    which is larger in *any* observed metric.
    """
    removed = np.zeros(len(df), dtype=bool)
    for target in target_metrics:
        for value in df[target].unique():
            group = (df[target] == value).to_numpy()
            if not (group & ~removed).any():
                continue
            best = df.loc[group & ~removed, 'fitness'].idxmax()
            lower_fitness = (df['fitness'] < df.at[best, 'fitness']).to_numpy()
            for col in observed_metrics:
                if col == 'fitness':
                    continue
                removed |= (df[col] > df.at[best, col]).to_numpy() & lower_fitness & group
    return ~removed


def pareto_mask(df: pd.DataFrame, metrics: list[str]) -> np.ndarray:
    """True for the rows no other row dominates in (fitness maximized, every metric minimized)."""
    points = np.c_[-df['fitness'].to_numpy(float), df[metrics].to_numpy(float)]
    p, q = points[:, None, :], points[None, :, :]
    return ~(np.all(q <= p, axis=2) & np.any(q < p, axis=2)).any(axis=1)


# --8<-- [end:filters]
# --8<-- [start:examples]
OBSERVED = ['switching_distance', 'disconnections']


def counterexample_a() -> pd.DataFrame:
    """A non-dominated row the filter removes: row 1 trades fitness for switching distance but adds a disconnection."""
    return pd.DataFrame(
        {
            'fitness': [-1.0, -2.0],
            'split_subs': [1, 1],
            'switching_distance': [3, 2],
            'disconnections': [0, 1],
        }
    )


def counterexample_b() -> pd.DataFrame:
    """Two dominated rows the filter keeps.

    Row 1 has the group best's metrics exactly and a worse fitness (dominated by row 0, kept because the filter asks
    for a strictly larger metric). Row 3 is dominated by row 2, which is not the group's best, and row 3 is better
    than the best (row 0) in switching distance, so the filter never looks at row 2.
    """
    return pd.DataFrame(
        {
            'fitness': [-1.0, -3.0, -2.0, -2.5],
            'split_subs': [1, 1, 1, 1],
            'switching_distance': [5, 5, 1, 2],
            'disconnections': [0, 0, 0, 0],
        }
    )


def order_example() -> pd.DataFrame:
    """Three rows whose mask depends on the order of the target metrics."""
    return pd.DataFrame(
        {
            'fitness': [-1.0, -2.0, -3.0],
            'split_subs': [1, 1, 2],
            'switching_distance': [2, 3, 3],
            'disconnections': [0, 1, 2],
        }
    )


# --8<-- [end:examples]
def main() -> None:
    """Both counterexamples and the order dependence."""
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    for name, df in (
        ('(a) non-dominated row removed', counterexample_a()),
        ('(b) dominated rows kept', counterexample_b()),
    ):
        print(name)
        print(df.assign(dominator=dominator_mask(df, ['split_subs'], OBSERVED), pareto=pareto_mask(df, OBSERVED)))
    df = order_example()
    first = dominator_mask(df, ['split_subs', 'switching_distance'], ['disconnections'])
    second = dominator_mask(df, ['switching_distance', 'split_subs'], ['disconnections'])
    print('order dependence')
    print(df.assign(split_subs_first=first, switching_distance_first=second, pareto=pareto_mask(df, OBSERVED)))


if __name__ == '__main__':
    main()
