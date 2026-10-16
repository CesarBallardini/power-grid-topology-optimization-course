import json

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import baseline_toop  # noqa: E402
import sld_toop  # noqa: E402


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    if not baseline_toop.NODE_BREAKER.exists():
        pytest.skip(f'{baseline_toop.NODE_BREAKER} not found')
    work_dir = baseline_toop.baseline(work_dir=tmp_path_factory.mktemp('capstone'), runs=1, runtime_seconds=5.0)
    return work_dir / 'run_0', work_dir / 'grid'


def test_before_and_after_diagrams_highlight_the_operated_switches(run):
    run_dir, data_folder = run
    summary = json.loads((run_dir / 'topology_0' / 'orao_summary.json').read_text(encoding='utf-8'))
    svgs, open_before, open_after = sld_toop.diagrams(
        data_folder / 'grid.xiidm', run_dir / 'topology_0' / 'modified_network.xiidm', summary
    )
    assert list(svgs) == ['VL2']  # the voltage level id; the station's name is VLevel2
    before, after = svgs['VL2']
    ids = [s for s, _ in sld_toop.operated_switches(summary)]
    assert ids == ['VL2_BREAKER', 'L12_DISCONNECTOR_4_0', 'L12_DISCONNECTOR_4_2']
    assert '<svg' in before[:300] and '<svg' in after[:300]
    assert sld_toop.highlighted(before) == len(ids) and sld_toop.highlighted(after) == len(ids)
    assert before.count('#0050fcff') == 7  # the colour sits in the CSS, once per voltage class
    assert open_before == {'VL2_BREAKER': False, 'L12_DISCONNECTOR_4_0': False, 'L12_DISCONNECTOR_4_2': True}
    assert open_after == {'VL2_BREAKER': True, 'L12_DISCONNECTOR_4_0': True, 'L12_DISCONNECTOR_4_2': False}
