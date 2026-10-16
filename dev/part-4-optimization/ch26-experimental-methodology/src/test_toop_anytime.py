import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from toop_anytime import EXAMPLE, collect, compact, per_seed, plot_anytime, truncate  # noqa: E402


@pytest.fixture(scope='module')
def seeds():
    return pd.read_csv(EXAMPLE)


def test_committed_runs(seeds):
    table = per_seed(seeds)
    assert table.index.tolist() == [1, 2, 3, 4, 5]
    assert table['epochs'].tolist() == [1307, 909, 1143, 1022, 1210]
    assert table['max_fitness'].tolist() == pytest.approx([-42.3453] * 5, abs=1e-3)
    assert table['first_epoch_at_max'].tolist() == [0] * 5
    assert table['total_branch_combis'].tolist() == [1045608, 727208, 914408, 817608, 968008]


def test_truncate_to_the_smallest_common_epoch(seeds):
    cut, common = truncate(seeds)
    assert common == 908 and cut['epoch'].max() == 908
    assert cut.groupby('seed')['total_branch_combis'].max().min() == 720808
    fig, ax = plt.subplots()
    plot_anytime(cut, 'total_branch_combis', ax)
    assert len(ax.get_lines()) == 5
    plt.close(fig)


def test_collect_and_compact_on_a_fake_folder(tmp_path):
    for seed, n_epochs in ((1, 5), (2, 3)):
        folder = tmp_path / f'seed{seed}' / 'results' / 'test'
        folder.mkdir(parents=True)
        for epoch in range(n_epochs):
            fitness = -10.0 if epoch < 2 else -5.0
            res = {
                'iteration': epoch,
                'max_fitness': fitness,
                'total_branch_combis': 100 * (epoch + 1),
                'args': {'ga_config': {'runtime_seconds': 1.0}},
            }
            (folder / f'res_{epoch}.json').write_text(json.dumps(res), encoding='utf-8')
    df = collect(tmp_path)
    assert len(df) == 8 and df['seed'].tolist() == [1] * 5 + [2] * 3
    small = compact(df, every=100)
    assert small[small['seed'] == 1]['epoch'].tolist() == [0, 2, 4]  # first, change, last
    assert small[small['seed'] == 2]['epoch'].tolist() == [0, 2]
    assert per_seed(df)['first_epoch_at_max'].tolist() == [2, 2]
