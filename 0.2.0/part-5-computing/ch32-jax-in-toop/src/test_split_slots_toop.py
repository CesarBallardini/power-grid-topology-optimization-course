import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

from solver_lab_toop import grid_folder, setup  # noqa: E402
from split_slots_toop import slot_sweep, unsplit_with_slots  # noqa: E402


@pytest.fixture(scope='module')
def si():
    return setup(grid_folder('case57'))


def test_unsplit_topologies_have_the_requested_slots(si):
    topologies = unsplit_with_slots(si.solver_config, 8, 3)
    assert topologies.action.shape == (8, 3) and bool(topologies.pad_mask.all())


def test_slot_sweep_reports_both_sizes(si):
    rows = slot_sweep(si, slots=(1, 4), n_topologies=32, repeats=3)
    assert [r['slots'] for r in rows] == [1, 4]
    assert all(r['steady_s'] > 0 and r['compile_s'] > 0 for r in rows)
    # On case57 the per-slot BSDF cost is small next to the N-1 analysis: no ordering is asserted here, the
    # script's table reports the measured ratio.
