"""From a node-breaker grid file to a search space: ToOp's import and preprocessing, stage by stage (Chapter 22 Lab).

Run it with ToOp's interpreter (or inside its container) from this folder::

    python grid_pipeline_toop.py /tmp/ch22

It copies ``data/grid_node_breaker/grid.xiidm`` into a working folder, imports it with ``convert_file``, preprocesses
it with ``load_grid`` while recording the size statistics of every stage, repeats the run with other preprocessing
parameters and import rules, and does the same for the generated complex grid of ``example3_e2e_pipeline.ipynb``.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pypowsybl as pp
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid
from toop_engine_grid_helpers.powsybl.example_grids import create_complex_grid_battery_hvdc_svc_3w_trafo
from toop_engine_grid_helpers.powsybl.powsybl_helpers import load_lf_params_from_fs
from toop_engine_importer.pypowsybl_import import preprocessing
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import (
    PreprocessParameters,
    ReassignmentLimits,
    RelevantStationRules,
)
from toop_engine_interfaces.status_update import empty_status_update_fn
from toop_engine_interfaces.stored_action_set import load_action_set
from toop_engine_topology_optimizer.benchmark.benchmark_utils import prepare_importer_parameters

TOOP_GRID = Path(__file__).resolve().parents[5] / 'ToOp' / 'data' / 'grid_node_breaker' / 'grid.xiidm'
STATS = ['n_nodes', 'n_branches', 'n_relevant_subs', 'n_actions', 'n_disc_branches']
VARIANTS = {
    'default': PreprocessParameters(),
    'action_set_filter_bsdf_lodf=False': PreprocessParameters(action_set_filter_bsdf_lodf=False),
    'action_set_filter_bridge_lookup=False': PreprocessParameters(action_set_filter_bridge_lookup=False),
    'action_set_clip=2**3': PreprocessParameters(action_set_clip=2**3),
    'max_reassignments_per_sub=1': PreprocessParameters(
        electrical_reassignment_limits=ReassignmentLimits(max_reassignments_per_sub=1)
    ),
}


def import_grid(
    work: Path,
    source: Path | pp.network.Network = TOOP_GRID,
    cutoff_voltage: float = 10,
    rules: RelevantStationRules | None = None,
) -> Path:
    """Step 1: copy (or save) the grid into ``work`` and run the importer; return the processed grid folder."""
    work.mkdir(parents=True, exist_ok=True)
    grid_file = work / 'grid.xiidm'
    if isinstance(source, pp.network.Network):
        source.save(str(grid_file))
    else:
        shutil.copy(source, grid_file)
    data_folder = work / 'grid'
    parameters = prepare_importer_parameters(grid_file, data_folder)
    parameters.area_settings.cutoff_voltage = cutoff_voltage
    if rules is not None:
        parameters.relevant_station_rules = rules
    preprocessing.convert_file(importer_parameters=parameters, status_update_fn=empty_status_update_fn)
    return data_folder


def mask_counts(data_folder: Path) -> pd.Series:
    """Step 2: number of ``True`` entries (or length, for weight masks) of every ``masks/*.npy``."""
    counts = {}
    for file in sorted((data_folder / PREPROCESSING_PATHS['masks_path']).glob('*.npy')):
        mask = np.load(file)
        counts[file.stem] = int(mask.sum()) if mask.dtype == bool else len(mask)
    return pd.Series(counts, name='true_entries')


def read_json(data_folder: Path, key: str) -> dict:
    """One of the JSON artifacts of the folder, by its ``PREPROCESSING_PATHS`` key."""
    with (data_folder / PREPROCESSING_PATHS[key]).open(encoding='utf-8') as f:
        return json.load(f)


def actions_per_station(data_folder: Path) -> pd.DataFrame:
    """Step 2: actions per ``bus_group_id`` of the action set against the bound 2^(d-1) - d (d branches).

    ``action_set.json`` holds the stations without their actions; the actions are stored as diffs in
    ``action_set_diffs.hdf5`` and expanded by ``load_action_set``.
    """
    paths = PREPROCESSING_PATHS
    action_set = load_action_set(
        data_folder / paths['action_set_file_path'], data_folder / paths['action_set_diff_path']
    )
    degree = {g.bus_group_id: len(g.branch_connections) for g in action_set.starting_bus_groups}
    counts = pd.Series([a.bus_group_id for a in action_set.local_actions]).value_counts()
    rows = [
        {'bus_group_id': k, 'branches': d, 'actions': int(counts.get(k, 0)), 'bound': 2 ** (d - 1) - d}
        for k, d in degree.items()
    ]
    return pd.DataFrame(rows).set_index('bus_group_id')


def funnel(data_folder: Path, parameters: PreprocessParameters | None = None) -> tuple[pd.DataFrame, dict, float]:
    """Step 3: ``load_grid`` with a recording callback; returns the stage table, the final stats and the seconds."""
    records = []

    def record(stage, message, *, stats=None):
        records.append({'stage': str(stage), **(stats or {})})

    lf_params = load_lf_params_from_fs(
        DirFileSystem(data_folder), Path(PREPROCESSING_PATHS['loadflow_parameters_file_path'])
    )
    jax.clear_caches()
    start = time.perf_counter()
    info, _, _ = load_grid(
        DirFileSystem(str(data_folder)),
        pandapower=False,
        parameters=parameters or PreprocessParameters(),
        status_update_fn=record,
        lf_params=lf_params,
    )
    seconds = time.perf_counter() - start
    table = pd.DataFrame(records).dropna(subset=['n_nodes']).set_index('stage')[STATS].astype(int)
    return table, dict(info), seconds


def stage_that_removes_most(table: pd.DataFrame) -> pd.Series:
    """Per statistic, the stage with the largest drop against the previous stage (``None`` when nothing drops)."""
    drops = table.diff()
    return pd.Series({c: (drops[c].idxmin() if drops[c].min() < 0 else None) for c in table.columns})


def variants(data_folder: Path) -> pd.DataFrame:
    """Step 4: ``n_actions`` and wall-clock time of the preprocessing under each parameter variant."""
    rows = []
    for label, parameters in VARIANTS.items():
        _, info, seconds = funnel(data_folder, parameters)
        rows.append(
            {
                'variant': label,
                'n_actions': info['n_actions'],
                'n_relevant_subs': info['n_relevant_subs'],
                'seconds': round(seconds, 1),
            }
        )
    return pd.DataFrame(rows).set_index('variant')


def import_rules(work: Path) -> pd.DataFrame:
    """Step 5: relevant substations after import and after preprocessing for three relevant-station rules."""
    rows = []
    for label, rules in (
        ('default (2 busbars, 4 branches)', None),
        ('min_busbars=3', RelevantStationRules(min_busbars=3)),
        ('min_connected_branches=2', RelevantStationRules(min_connected_branches=2)),
    ):
        data_folder = import_grid(work / label.split(' ')[0].replace('=', ''), rules=rules)
        after_import = read_json(data_folder, 'importer_auxiliary_file_path')['id_lists']['relevant_subs']
        _, info, _ = funnel(data_folder)
        rows.append(
            {
                'rule': label,
                'after_import': after_import,
                'after_preprocessing': info['n_relevant_subs'],
                'n_actions': info['n_actions'],
            }
        )
    return pd.DataFrame(rows).set_index('rule')


def main() -> None:
    """Steps 1 to 6 of the Lab, printing every number the solutions quote."""
    pd.set_option('display.width', 200)
    work = Path(sys.argv[1] if len(sys.argv) > 1 else 'ch22')
    data_folder = import_grid(work / 'node_breaker')
    print('step 1: processed grid folder written:', sorted(p.name for p in data_folder.iterdir()))
    print('step 2: masks with True entries:')
    counts = mask_counts(data_folder)
    print(counts[counts > 0].to_string())
    aux = read_json(data_folder, 'importer_auxiliary_file_path')['id_lists']
    print('        relevant_subs after import:', aux['relevant_subs'])
    nminus1 = read_json(data_folder, 'nminus1_definition_file_path')
    n_mon, n_con = len(nminus1['monitored_elements']), len(nminus1['contingencies'])
    print(f'        N-1 definition: {n_mon} monitored elements, {n_con} contingencies')
    table, info, seconds = funnel(data_folder)
    print('        actions per station:')
    print(actions_per_station(data_folder).to_string())
    keys = (
        'n_relevant_subs',
        'n_actions',
        'n_monitored_branches',
        'n_nminus1_cases',
        'overload_energy_n0',
        'overload_energy_n1',
    )
    print('        static_information_stats.json:', {k: info[k] for k in keys})
    print(f'step 3: the funnel ({seconds:.1f} s):')
    print(table.to_string())
    print('        stage with the largest drop:', stage_that_removes_most(table).to_dict())
    print('step 4: preprocessing variants:')
    print(variants(data_folder).to_string())
    print('step 5: import rules:')
    print(import_rules(work / 'rules').to_string())
    complex_folder = import_grid(work / 'complex', create_complex_grid_battery_hvdc_svc_3w_trafo())
    table_c, info_c, seconds_c = funnel(complex_folder)
    relevant = read_json(complex_folder, 'importer_auxiliary_file_path')['id_lists']['relevant_subs']
    print(f'step 6: complex grid ({seconds_c:.1f} s), {len(relevant)} relevant substations after import')
    print(table_c.to_string())
    keys = ('n_nodes', 'n_branches', 'n_relevant_subs', 'n_actions', 'n_disc_branches', 'n_controllable_psts')
    print('        stats:', {k: info_c[k] for k in (*keys, 'overload_energy_n0', 'overload_energy_n1')})
    for cutoff in (220, 400):
        folder = import_grid(
            work / f'complex_{cutoff}', create_complex_grid_battery_hvdc_svc_3w_trafo(), cutoff_voltage=cutoff
        )
        relevant = read_json(folder, 'importer_auxiliary_file_path')['id_lists']['relevant_subs']
        print(f'        cutoff {cutoff} kV: {len(relevant)} relevant substations after import: {relevant}')
        try:
            _, info_cut, _ = funnel(folder)
            n_subs, n_actions = info_cut['n_relevant_subs'], info_cut['n_actions']
            print(f'        preprocessed: {n_subs} relevant substations, {n_actions} actions')
        except Exception as error:  # noqa: BLE001 - the point is to show how an empty search space fails
            print(f'        preprocessing fails: {type(error).__name__}: {error}')


if __name__ == '__main__':
    main()
