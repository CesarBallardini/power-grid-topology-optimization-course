"""ToOp's own acceptance verdicts on the stored Lab B run, and the diagram helpers (`make test-toop`)."""

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
import handoff  # noqa: E402
import lab_b_toop  # noqa: E402
from acceptance import run_table  # noqa: E402


@pytest.mark.parametrize('grid', ['node_breaker', 'complex'])
def test_toop_verdicts_match_the_port_on_the_stored_run(grid):
    run_dir = handoff.RUNS / grid
    table = run_table(run_dir)
    verdicts = lab_b_toop.toop_verdicts(run_dir)
    assert dict(zip(table['topology'], table['rejected_by'], strict=True)) == verdicts


def test_highlight_counts_classes_not_the_colour():
    svg = '<style>.a.highlight {--c: #0050fcff}</style><g class="sld-breaker highlight"/><g class="sld-wire"/>'
    assert lab_b_toop.highlighted(svg) == 1


def test_operated_switches_reads_the_summary():
    summary = handoff.load(handoff.RUNS / 'node_breaker' / 'topology_0' / 'orao_summary.json')
    assert lab_b_toop.operated_switches(summary) == [
        ('VL2_BREAKER', True),
        ('L12_DISCONNECTOR_4_0', True),
        ('L12_DISCONNECTOR_4_2', False),
    ]
