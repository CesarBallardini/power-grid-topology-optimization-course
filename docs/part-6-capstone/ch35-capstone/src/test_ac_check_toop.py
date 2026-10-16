import json

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import ac_check_toop  # noqa: E402
import baseline_toop  # noqa: E402


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    if not baseline_toop.NODE_BREAKER.exists():
        pytest.skip(f'{baseline_toop.NODE_BREAKER} not found')
    work_dir = baseline_toop.baseline(work_dir=tmp_path_factory.mktemp('capstone'), runs=1, runtime_seconds=5.0)
    return work_dir / 'run_0', work_dir / 'grid'


def test_runner_reproduces_the_stored_metrics_and_pypowsybl_agrees_on_n0(run):
    run_dir, data_folder = run
    rows = ac_check_toop.compare(run_dir, data_folder)
    assert len(rows) == 1
    row = rows[0]
    assert row['largest_metric_gap'] < 1e-9
    assert row['file_fitness'] == pytest.approx(row['runner_fitness'])
    assert row['pypowsybl_max_loading_n_0'] == pytest.approx(row['file_max_flow_n_0'], abs=1e-3)


def test_n1_route_matches_the_files_only_with_the_stored_load_flow_parameters(run):
    run_dir, data_folder = run
    unsplit = json.loads((run_dir / 'unsplit_ac_metrics.json').read_text(encoding='utf-8'))['extra_scores']
    stored = ac_check_toop.n1_route(
        data_folder, data_folder / 'grid.xiidm', ac_check_toop.stored_lf_params(data_folder)
    )
    assert stored['overload_energy_n_1'] == pytest.approx(unsplit['overload_energy_n_1'], abs=1e-9)
    assert stored['non_converging'] == unsplit['non_converging_loadflows'] == 7
    assert stored['not_converged'] == ['BBS2_1', 'BBS3_1', 'BBS3_2', 'BBS5_1', 'generator3', 'load1', 'load2']
    default = ac_check_toop.n1_route(data_folder, data_folder / 'grid.xiidm')
    assert default['non_converging'] == 0 and default['overload_energy_n_1'] > stored['overload_energy_n_1']


def test_metrics_keyword_is_silently_dropped_in_two_lines():
    from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

    metrics = Metrics(fitness=0.0, extra_scores={}, worst_k_contingent_cases=['L8'])
    assert metrics.worst_k_contingency_cases is None


@pytest.mark.xfail(
    strict=True, reason='ba1874e: compute_metrics_single_timestep passes worst_k_contingent_cases to Metrics'
)
def test_ac_metrics_carry_the_worst_contingencies(run):
    run_dir, data_folder = run
    stored = json.loads((run_dir / 'topology_0' / 'ac_metrics.json').read_text(encoding='utf-8'))
    rebuilt = ac_check_toop.runner_metrics(
        data_folder, data_folder / 'grid.xiidm', stored['dc_info']['actions'], stored['dc_info']['disconnections']
    )
    assert rebuilt.worst_k_contingency_cases is not None
