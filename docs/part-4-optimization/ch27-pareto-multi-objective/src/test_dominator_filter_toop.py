"""Compares the ports with ToOp's own functions; runs only where ToOp is importable (`make test-toop`)."""

import numpy as np
import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from dominator_filter import OBSERVED, counterexample_a, counterexample_b, dominator_mask, order_example  # noqa: E402
from repertoire_cells import N_CELLS, cell_index  # noqa: E402
from toop_engine_topology_optimizer.ac.select_strategy import get_dominator_mask  # noqa: E402
from toop_engine_topology_optimizer.dc.repertoire.discrete_me_repertoire import get_cell_index  # noqa: E402


@pytest.mark.parametrize('frame', [counterexample_a, counterexample_b, order_example])
@pytest.mark.parametrize(
    'targets', [['split_subs'], ['split_subs', 'switching_distance'], ['switching_distance', 'split_subs']]
)
def test_port_matches_toop(frame, targets):
    df = frame()
    np.testing.assert_array_equal(dominator_mask(df, targets, OBSERVED), get_dominator_mask(df, targets, OBSERVED))
    observed = ['disconnections']
    np.testing.assert_array_equal(dominator_mask(df, targets, observed), get_dominator_mask(df, targets, observed))


def test_toop_removes_a_non_dominated_row_and_keeps_dominated_ones():
    assert get_dominator_mask(counterexample_a(), ['split_subs'], OBSERVED).tolist() == [True, False]
    assert get_dominator_mask(counterexample_b(), ['split_subs'], OBSERVED).all()


@pytest.mark.parametrize('descriptor', [(1, 2), (2, 5), (4, 44), (4, 60), (7, 44), (0, 0)])
def test_cell_index_matches_toop(descriptor):
    assert cell_index(descriptor) == int(get_cell_index(np.array(descriptor), N_CELLS))
