"""Runs only where ToOp is importable (its container or virtual environment, `make test-toop`); skipped elsewhere."""

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_contingency_analysis')
import pypowsybl  # noqa: E402
from ac_n1_toop import (  # noqa: E402
    BASECASE,
    default_grid_path,
    expected_rows,
    frames,
    n1_definition,
    pandas_metrics,
    run,
)
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics  # noqa: E402


@pytest.fixture(scope='module')
def results():
    path = default_grid_path()
    if not path.exists():
        pytest.skip(f'{path} not found: set TOOP_REPO')
    net = pypowsybl.network.load(str(path))
    definition = n1_definition(net)
    res = run(net, definition)
    return net, definition, res, frames(res)


def test_definition_covers_every_branch_and_bus(results):
    net, definition, _, _ = results
    assert len(definition.monitored_elements) == len(net.get_lines()) + len(net.get_buses())
    assert len(definition.contingencies) == len(net.get_lines()) + 1
    assert definition.contingencies[0].id == BASECASE and definition.contingencies[0].elements == []


def test_row_count_is_below_the_formula_by_the_outaged_branches(results):
    net, definition, _, f = results
    branch = f['branch']
    assert (f['converged'].status == 'CONVERGED').all()
    assert len(branch) == expected_rows(net, len(definition.contingencies)) - 2 * len(net.get_lines())
    assert (branch.contingency == branch.element).sum() == 0


def test_pandas_metrics_match_compute_metrics(results):
    _, _, res, f = results
    mine = pandas_metrics(f['branch'], f['va_diff'])
    toop = compute_metrics(res, base_case_id=BASECASE)
    for name, value in mine.items():
        assert value == pytest.approx(toop[name], abs=1e-9), name
    assert mine['critical_branch_count_n_1'] == 3
