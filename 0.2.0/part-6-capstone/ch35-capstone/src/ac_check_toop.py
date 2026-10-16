"""Rebuild the AC N-1 result of a validated topology and compare it with ``ac_metrics.json`` (Chapter 35, Lab step 3).

Three checks. ``runner_metrics`` goes the way the lab allows,
``create_loadflow_runner(data_folder, grid_file_path).run_ac_loadflow(actions, disconnections)`` followed by
``compute_metrics_single_timestep``, which is ToOp's own code path and must reproduce the file exactly.
``independent_n0`` loads ``topology_N/modified_network.xiidm`` with pypowsybl alone, runs an AC load flow and
computes the largest current loading from the permanent limits: an N-0 figure that owes nothing to ToOp, to be
compared with ``max_flow_n_0``. ``n1_route`` is the lab's first route: the preprocessed ``nminus1_definition.json``
(its monitored elements are ``MonitoredElement`` objects) and ``get_ac_loadflow_results`` on the modified network,
with the overload energy summed again in pandas rather than by ``compute_metrics``, once with pypowsybl's default
load-flow parameters and once with the preprocessed ``loadflow_parameters`` that the runner loads. Run with ToOp's
interpreter: ``python ac_check_toop.py <run folder> <data folder>``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pypowsybl as pp
from fsspec.implementations.dirfs import DirFileSystem
from omegaconf import DictConfig
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.nminus1_definition import load_nminus1_definition
from toop_engine_topology_optimizer.ac.scoring_functions import compute_metrics_single_timestep
from toop_engine_topology_optimizer.benchmark.benchmark_utils import create_loadflow_runner

N1_KEYS = ('overload_energy_n_1', 'critical_branch_count_n_1', 'non_converging_loadflows')
AC_CONFIG = DictConfig({'critical_voltage_jump_percent': 5.0, 'critical_va_diff_degree': 20.0})


def runner_metrics(data_folder, grid_xiidm, actions, disconnections, ac_cfg=AC_CONFIG):
    """ToOp's AC metrics of one topology through the load-flow runner (the lab's second route)."""
    runner = create_loadflow_runner(Path(data_folder), Path(grid_xiidm), n_processes=1)
    loadflow = runner.run_ac_loadflow(list(actions), list(disconnections))
    info = runner.get_last_action_info()
    definition = runner.get_nminus1_definition()
    base_case_id = definition.base_case.id if definition.base_case is not None else None
    return compute_metrics_single_timestep(
        actions=list(actions),
        disconnections=list(disconnections),
        loadflow=loadflow,
        additional_info=info,
        base_case_id=base_case_id,
        critical_voltage_jump_percent=ac_cfg['critical_voltage_jump_percent'],
        critical_va_diff_degree=ac_cfg['critical_va_diff_degree'],
    )


def independent_n0(network_xiidm):
    """Largest N-0 current loading (current over permanent limit) of the lines, with pypowsybl only.

    Lines only: ToOp's ``max_flow_n_0`` also covers transformers, whose current limits need the side convention of
    the monitored element, so the two agree on a grid whose worst branch is a line (the node-breaker example) and
    not on one whose worst element is a transformer (the complex example grid).
    """
    net = pp.network.load(str(network_xiidm))
    pp.loadflow.run_ac(net)
    branches = net.get_lines()[['i1', 'i2']]
    limits = net.get_operational_limits().reset_index()  # type and acceptable_duration are index levels
    permanent = limits[(limits['type'] == 'CURRENT') & (limits['acceptable_duration'] == -1)]
    loadings = {}
    for element_id, group in permanent.groupby('element_id'):
        if element_id not in branches.index:
            continue
        current = max(abs(branches.loc[element_id, 'i1']), abs(branches.loc[element_id, 'i2']))
        loadings[element_id] = current / group['value'].min()
    worst = max(loadings, key=loadings.get)
    return loadings[worst], worst


def stored_lf_params(data_folder):
    """The load-flow parameters the preprocessing stored, which ``create_loadflow_runner`` passes to every run."""
    return load_lf_params_from_fs(
        DirFileSystem(str(data_folder)), Path(PREPROCESSING_PATHS['loadflow_parameters_file_path'])
    )


def n1_route(data_folder, network_xiidm, lf_params=None):
    """The AC N-1 of a network with ToOp's N-1 definition, outside the runner; metrics recomputed with pandas.

    ``lf_params=None`` is pypowsybl's default; ``stored_lf_params(data_folder)`` is what the runner uses. Per
    monitored branch the worst contingency and side counts, its overload being |p| - |p| / loading; branches
    with loading above 1 are critical; contingencies whose status is not ``CONVERGED`` are non-converging.
    """
    definition = load_nminus1_definition(Path(data_folder) / 'nminus1_definition.json')
    net = pp.network.load(str(network_xiidm))
    results = get_ac_loadflow_results(
        net, definition, job_id='capstone', timestep=0, n_processes=1, lf_params=lf_params
    )
    branch = results.branch_results.collect().to_pandas()
    converged = results.converged.collect().to_pandas()
    base = definition.base_case.id if definition.base_case is not None else None
    n1 = branch[branch.contingency != base]
    over = n1['p'].abs() - (n1['p'] / n1['loading']).abs()
    worst = over.groupby([n1.timestep, n1.element]).max()
    critical = n1[n1.loading.fillna(-1) > 1.0][['timestep', 'element']].drop_duplicates()
    return {
        'contingencies': len(definition.contingencies),
        'monitored_elements': len(definition.monitored_elements),
        'overload_energy_n_1': float(worst[worst > 0].sum()),
        'critical_branch_count_n_1': float(len(critical)),
        'non_converging': int((converged.status != 'CONVERGED').sum()),
        'statuses': converged.status.value_counts().to_dict(),
        'not_converged': sorted(converged.loc[converged.status != 'CONVERGED', 'contingency'].unique()),
    }


def compare(run_dir, data_folder):
    """Per validated topology: the file's metrics, the runner's and pypowsybl's N-0 loading."""
    run_dir, data_folder = Path(run_dir), Path(data_folder)
    rows = []
    for path in sorted(run_dir.glob('topology_*/ac_metrics.json')):
        stored = json.loads(path.read_text(encoding='utf-8'))
        actions, disconnections = stored['dc_info']['actions'], stored['dc_info']['disconnections']
        rebuilt = runner_metrics(data_folder, data_folder / 'grid.xiidm', actions, disconnections)
        n0, worst = independent_n0(path.parent / 'modified_network.xiidm')
        n1 = n1_route(data_folder, path.parent / 'modified_network.xiidm', stored_lf_params(data_folder))
        shared = [k for k in stored['extra_scores'] if k in rebuilt.extra_scores]
        gap = max(abs(stored['extra_scores'][k] - rebuilt.extra_scores[k]) for k in shared)
        rows.append(
            {
                'topology': path.parent.name,
                'actions': actions,
                'file_fitness': stored['fitness'],
                'runner_fitness': rebuilt.fitness,
                'largest_metric_gap': gap,
                'file_max_flow_n_0': stored['extra_scores']['max_flow_n_0'],
                'pypowsybl_max_loading_n_0': n0,
                'pypowsybl_worst_line': worst,
                'runner_worst_k': rebuilt.worst_k_contingency_cases,
                'file_overload_energy_n_1': stored['extra_scores']['overload_energy_n_1'],
                'n1_route_overload_energy_n_1': n1['overload_energy_n_1'],
                'file_critical_branch_count_n_1': stored['extra_scores']['critical_branch_count_n_1'],
                'n1_route_critical_branch_count_n_1': n1['critical_branch_count_n_1'],
                'file_non_converging': stored['extra_scores']['non_converging_loadflows'],
                'n1_route_non_converging': n1['non_converging'],
            }
        )
    return rows


def main():
    """Compare every validated topology of a run folder against the preprocessed data folder."""
    run_dir, data_folder = Path(sys.argv[1]), Path(sys.argv[2])
    for row in compare(run_dir, data_folder):
        print(row)
    unsplit = json.loads((run_dir / 'unsplit_ac_metrics.json').read_text(encoding='utf-8'))
    n0, worst = independent_n0(data_folder / 'grid.xiidm')
    print(f'unsplit: file max_flow_n_0 {unsplit["extra_scores"]["max_flow_n_0"]:.4f}, pypowsybl {n0:.4f} on {worst}')
    print('unsplit file:', {k: unsplit['extra_scores'][k] for k in N1_KEYS})
    print('unsplit N-1 route, default parameters:', n1_route(data_folder, data_folder / 'grid.xiidm'))
    stored = n1_route(data_folder, data_folder / 'grid.xiidm', stored_lf_params(data_folder))
    print('unsplit N-1 route, stored parameters:', stored)
    print('agreement on N-0 within 1e-3:', bool(np.isclose(unsplit['extra_scores']['max_flow_n_0'], n0, atol=1e-3)))


if __name__ == '__main__':
    main()
