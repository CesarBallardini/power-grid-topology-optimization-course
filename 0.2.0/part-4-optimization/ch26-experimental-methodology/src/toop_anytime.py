"""Anytime curves of repeated ToOp runs from their ``res_<epoch>.json`` files (Chapter 26, Lab step 6).

ToOp writes one ``res_<epoch>.json`` per epoch into ``stats_dir``; each holds ``max_fitness``, ``iteration`` and
the cumulative counter ``total_branch_combis``. ``collect`` reads a folder with one run per seed (as written by
Chapter 25's ``small_grid_toop.py OUT seed<k> --seed <k>``), ``compact`` keeps only the epochs where something
changes so that the table can be committed (``toop_seeds.csv``), ``truncate`` cuts every run at the smallest
common epoch, and ``plot_anytime`` draws best fitness against epochs or against evaluated topologies.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / 'toop_seeds.csv'


def collect(folder: Path) -> pd.DataFrame:
    """One row per (seed, epoch) from ``folder/seed<k>/results/*/res_<epoch>.json``."""
    rows = []
    for run in sorted(folder.glob('seed*')):
        seed = int(run.name.removeprefix('seed'))
        for path in run.glob('results/*/res_*.json'):
            res = json.loads(path.read_text(encoding='utf-8'))
            rows.append(
                {
                    'seed': seed,
                    'epoch': res['iteration'],
                    'max_fitness': res['max_fitness'],
                    'total_branch_combis': res['total_branch_combis'],
                    'runtime_seconds': res['args']['ga_config']['runtime_seconds'],
                }
            )
    return pd.DataFrame(rows).sort_values(['seed', 'epoch']).reset_index(drop=True)


def compact(df: pd.DataFrame, every: int = 100) -> pd.DataFrame:
    """Keep the first epoch, every epoch where ``max_fitness`` changes, every ``every``-th epoch and the last one."""
    keep = []
    for _, run in df.groupby('seed'):
        changed = run['max_fitness'].diff().fillna(1.0) != 0
        periodic = run['epoch'] % every == 0
        last = run['epoch'] == run['epoch'].max()
        keep.append(run[changed | periodic | last])
    return pd.concat(keep).reset_index(drop=True)


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Epochs completed, topologies evaluated, final best fitness and the first epoch at which it was reached."""
    rows = []
    for seed, run in df.groupby('seed'):
        final = run['max_fitness'].max()
        rows.append(
            {
                'seed': seed,
                'epochs': int(run['epoch'].max()) + 1,
                'total_branch_combis': int(run['total_branch_combis'].max()),
                'max_fitness': final,
                'first_epoch_at_max': int(run.loc[run['max_fitness'] >= final - 1e-9, 'epoch'].min()),
            }
        )
    return pd.DataFrame(rows).set_index('seed')


def truncate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows up to the smallest number of epochs any seed completed: an equal budget for every run."""
    common = int(df.groupby('seed')['epoch'].max().min())
    return df[df['epoch'] <= common].reset_index(drop=True), common


def plot_anytime(df: pd.DataFrame, x: str = 'epoch', ax=None) -> None:
    """Best fitness so far against ``x`` (``epoch`` or ``total_branch_combis``), one step line per seed."""
    ax = ax or plt.gca()
    for seed, run in df.groupby('seed'):
        ax.step(run[x], run['max_fitness'], where='post', label=f'seed {seed}')
    ax.set_xlabel(x.replace('_', ' '))
    ax.set_ylabel('best fitness')
    ax.legend()


def main() -> None:
    """Collect and compact a folder of runs into ``toop_seeds.csv`` (one argument), then summarize that table."""
    if len(sys.argv) == 2:
        compact(collect(Path(sys.argv[1]))).to_csv(EXAMPLE, index=False)
        print(f'wrote {EXAMPLE}')
    df = pd.read_csv(EXAMPLE)
    print(
        f'{df["seed"].nunique()} seeds, runtime {df["runtime_seconds"].iloc[0]:.0f} s each, {len(df)} compacted rows'
    )
    table = per_seed(df)
    print(table.to_string())
    spread = table['epochs'].max() / table['epochs'].min()
    print(f'epochs: min {table["epochs"].min()}, max {table["epochs"].max()} (ratio {spread:.2f})')
    cut, common = truncate(df)
    by_seed = cut.groupby('seed')
    print(
        f'equal budget: {common + 1} epochs, {int(by_seed["total_branch_combis"].max().min())} topologies; '
        f'best fitness per seed {by_seed["max_fitness"].max().round(3).tolist()}'
    )


if __name__ == '__main__':
    main()
