import json

import numpy as np
import pytest
from repertoire_cells import (
    METRICS,
    N_CELLS,
    analyse,
    cell_index,
    exact_cell_index,
    synthetic_repertoire,
    table_from_res_json,
)


def test_exercise_5_cells():
    assert cell_index((1, 2)) == 47 and exact_cell_index((1, 2)) == 47
    assert cell_index((4, 60)) == cell_index((4, 44)) == 224
    assert exact_cell_index((4, 60)) is None and exact_cell_index((7, 44)) is None
    assert cell_index((7, 60)) == np.prod(N_CELLS) - 1


def test_synthetic_repertoire_has_one_elite_per_cell():
    table = synthetic_repertoire()
    assert table['cell'].is_unique
    result = analyse(table)
    assert result['dominated_count'] + len(result['non_dominated']) == len(table)
    assert result['dominated_count'] == 19 and len(result['non_dominated']) == 11
    assert result['best_fitness'] in result['non_dominated']  # the weighted-sum winner is a supported point
    assert len(result['edge_cell_elites']) == 8


def test_table_from_res_json(tmp_path):
    topo = {'metrics': {'fitness': -1.5, 'extra_scores': dict.fromkeys(METRICS, 2.0)}}
    path = tmp_path / 'res.json'
    path.write_text(json.dumps({'best_topos': [topo, topo]}), encoding='utf-8')
    table = table_from_res_json(path)
    assert list(table.columns) == ['fitness', *METRICS] and len(table) == 2
    assert table['fitness'].iloc[0] == pytest.approx(-1.5)
