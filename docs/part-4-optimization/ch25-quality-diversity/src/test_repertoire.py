import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from repertoire import (  # noqa: E402
    EXAMPLE,
    cells_with_value,
    clipped_cells,
    coverage,
    descriptors,
    elites,
    grid,
    heatmap,
    load,
    qd_score,
    trim,
)


@pytest.fixture(scope='module')
def res():
    return load(EXAMPLE)


def test_committed_run_is_the_two_descriptor_step(res):
    assert descriptors(res) == [('split_subs', 4), ('switching_distance', 20)]
    assert res['initial_fitness'] == pytest.approx(-61.4446, abs=1e-3)
    assert res['max_fitness'] == pytest.approx(-42.3453, abs=1e-3)
    table = elites(res)
    assert len(table) == 3 and table['split_subs'].tolist() == [1.0, 1.0, 1.0]
    assert sorted(table['switching_distance']) == [0.0, 2.0, 4.0]


def test_grid_coverage_and_qd_score(res):
    table = grid(res)
    assert table.shape == (4, 20) and int(np.isfinite(table).sum()) == 3
    assert table[1, 2] == pytest.approx(-42.3453, abs=1e-3) and np.isnan(table[0, 0])
    assert coverage(table) == pytest.approx(3 / 80)
    assert qd_score(table, res['initial_fitness']) == pytest.approx(27.873, abs=1e-3)
    fig, ax = plt.subplots()
    heatmap(table, descriptors(res), ax)
    assert len(ax.images) == 1
    plt.close(fig)


def test_trim_is_idempotent_and_keeps_the_fields_used(res):
    again = trim(res)
    assert again == res
    assert set(again) == {
        'initial_fitness',
        'initial_metrics',
        'max_fitness',
        'iteration',
        'total_branch_combis',
        'args',
        'best_topos',
    }


def test_cells_of_the_default_repertoire():
    dims = (5, 45)
    two = cells_with_value(dims, 0, 2)
    assert two == list(range(90, 135))
    assert cells_with_value(dims, 0, 7) == list(range(180, 225))  # 7 splits are clipped into the last row
    clipped = clipped_cells(dims)
    assert len(clipped) == 45 + 5 - 1 == 49
    assert set(range(180, 225)) <= set(clipped) and {44, 89, 134, 179, 224} <= set(clipped)
