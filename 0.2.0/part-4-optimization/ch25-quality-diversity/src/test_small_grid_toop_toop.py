"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from small_grid_toop import GRID, ga_config, parse_pairs, run, summary  # noqa: E402


def test_ga_config_adds_descriptors_and_targets_to_the_observed_metrics():
    cfg = ga_config(
        'split_subs:4,switching_distance:20', 'overload_energy_n_1:1,critical_branch_count_n_1:10', 7, 5, True
    )
    assert cfg['me_descriptors'] == [
        {'metric': 'split_subs', 'num_cells': 4},
        {'metric': 'switching_distance', 'num_cells': 20},
    ]
    assert cfg['target_metrics'] == [['overload_energy_n_1', 1.0], ['critical_branch_count_n_1', 10.0]]
    assert cfg['observed_metrics'] == [
        'overload_energy_n_1',
        'split_subs',
        'switching_distance',
        'critical_branch_count_n_1',
    ]
    assert cfg['random_seed'] == 7 and cfg['runtime_seconds'] == 5 and cfg['plot'] is True
    assert parse_pairs('a:1,b:2') == [('a', 1), ('b', 2)]


def test_five_second_run_writes_res_json(tmp_path):
    assert GRID.exists(), f'ToOp example grid not found at {GRID}'
    run_dir = run(tmp_path, 'quick', ga_config('split_subs:4', 'overload_energy_n_1:1', 42, 5.0, False))
    res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
    s = summary(run_dir)
    assert s['initial_fitness'] == pytest.approx(-61.4446, abs=1e-3)  # the unsplit small grid, 61.4 MW
    assert s['max_fitness'] >= s['initial_fitness'] and s['epochs'] >= 1 and s['reported_elites'] >= 1
    assert s['descriptors'] == [('split_subs', 4)]
    assert set(res) >= {'best_topos', 'initial_metrics', 'args', 'iteration', 'total_branch_combis'}
    assert res['best_topos'][0]['metrics']['extra_scores']['split_subs'] == 1.0
    assert sorted((tmp_path / 'quick' / 'results' / 'test').glob('res_*.json'))  # one file per epoch


def test_plot_repertoire_refuses_three_descriptors(tmp_path):
    import jax.numpy as jnp
    from toop_engine_topology_optimizer.dc.repertoire.plotting import plot_repertoire

    fitnesses = jnp.full(5 * 45 * 3, -jnp.inf)
    with pytest.raises(ValueError, match='Only 1D and 2D'):
        plot_repertoire(fitnesses, None, str(tmp_path), (5, 45, 3), ('a', 'b', 'c'), save_plot=False)
    assert np.prod((5, 45, 3)) == 675
