"""The processed grid folder: who writes and reads each file, and what ``static_information.hdf5`` holds (Chapter 34).

Lab step 1 and Exercise 2. Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe folder_contract_toop.py [processed folder]

The default folder is ToOp's tracked ``data/case57_data_powsybl``. The writer/reader table was compiled from the
uses of each ``PREPROCESSING_PATHS`` key in ToOp's packages at commit ``ba1874e``.
"""

import dataclasses
import os
import sys
from pathlib import Path

import h5py
import pandas as pd
from toop_engine_dc_solver.jax.inputs import load_static_information
from toop_engine_dc_solver.jax.types import ActionSet, DynamicInformation, SolverConfig
from toop_engine_interfaces.folder_structure import NETWORK_MASK_NAMES, PREPROCESSING_PATHS

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
CASE57 = TOOP / 'data' / 'case57_data_powsybl'

# key of PREPROCESSING_PATHS -> (stage that writes it, stages that read it); "importer" is the pypowsybl import,
# "preprocessing" is dc_solver.preprocess (convert_to_jax, the backends, write_aux_data), "DC" the DC optimizer,
# "AC" the AC worker and the benchmark's load-flow runner.
STAGES = {
    'grid_file_path_powsybl': ('importer', 'preprocessing (powsybl_backend), AC (optimizer, loadflow runner)'),
    'grid_file_path_pandapower': ('importer (pp_masks)', 'preprocessing (pandapower_backend), AC'),
    'masks_path': ('importer (powsybl_masks, pp_masks)', 'preprocessing (both backends)'),
    'static_information_file_path': (
        'preprocessing (convert_to_jax)',
        'DC (main.fixed_files, worker via GridFile), benchmark',
    ),
    'static_information_stats_file_path': ('preprocessing (convert_to_jax)', 'nobody in the code'),
    'action_set_file_path': ('preprocessing (postprocess/write_aux_data)', 'AC (create_loadflow_runner, GridFile)'),
    'action_set_diff_path': ('preprocessing (postprocess/write_aux_data)', 'AC (create_loadflow_runner, GridFile)'),
    'nminus1_definition_file_path': ('importer, preprocessing (write_aux_data)', 'importer worker, AC runner'),
    'loadflow_parameters_file_path': ('importer', 'preprocessing (run_preprocessing, preprocessor), AC runner'),
    'importer_auxiliary_file_path': ('importer', 'nobody through the constant'),
    'asset_topology_master_data_file_path': (
        'importer, preprocessing (powsybl_backend)',
        'pandapower_backend, example grids',
    ),
    'original_gridfile_path': ('importer', 'nobody through the constant'),
    'asset_topology_file_path': ('an older preprocessing (no writer through the constant at ba1874e)', 'nobody'),
    'start_datetime_info_file_path': ('preprocessing backends', 'example grids'),
    'logs_path': ('preprocessing backends', 'example grids'),
}


def list_folder(folder=CASE57):
    """One row per file of a processed folder: its ``PREPROCESSING_PATHS`` key, writer and readers."""
    by_path = {v: k for k, v in PREPROCESSING_PATHS.items()}
    masks = {v: k for k, v in NETWORK_MASK_NAMES.items()}
    rows = []
    for path in sorted(p for p in Path(folder).rglob('*') if p.is_file()):
        rel = path.relative_to(folder).as_posix()
        key = by_path.get(rel)
        if key is None and rel.startswith(PREPROCESSING_PATHS['masks_path'] + '/'):
            key = 'masks_path' if path.name in masks else None
        parent_key = by_path.get(rel.rsplit('/', 1)[0]) if '/' in rel else None
        key = key or parent_key
        writer, readers = STAGES.get(key, ('?', '?'))
        rows.append({'file': rel, 'key': key, 'writer': writer, 'readers': readers})
    return pd.DataFrame(rows)


def hdf5_summary(path):
    """Dataset names with shapes and dtypes, and the attributes, of a static information file."""
    with h5py.File(path, 'r') as file:
        datasets = {name: (tuple(file[name].shape), str(file[name].dtype)) for name in file}
        attrs = {k: (v.item() if hasattr(v, 'item') else v) for k, v in file.attrs.items()}
    return datasets, attrs


def field_mapping(path):
    """Which ``DynamicInformation``, ``ActionSet`` or ``SolverConfig`` field each dataset and attribute maps to."""
    datasets, attrs = hdf5_summary(path)
    dyn = {f.name for f in dataclasses.fields(DynamicInformation)}
    act = {f.name for f in dataclasses.fields(ActionSet)}
    cfg = {f.name for f in dataclasses.fields(SolverConfig)}
    mapping = {}
    for name in datasets:
        if name in dyn:
            mapping[name] = f'DynamicInformation.{name}'
        elif name.startswith('action_set_') and name.removeprefix('action_set_') in act:
            mapping[name] = f'DynamicInformation.action_set.{name.removeprefix("action_set_")}'
        elif name in cfg:
            mapping[name] = f'SolverConfig.{name} (a dataset, not an attribute)'
        else:
            mapping[name] = 'built into another field at load time'
    attr_map = {k: (f'SolverConfig.{k}' if k in cfg else 'file metadata') for k in attrs}
    return mapping, attr_map


def main():
    """Print the folder table, the PTDF shape and the dataset/attribute mapping of the chosen folder."""
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else CASE57
    table = list_folder(folder)
    print(table.to_string(index=False))
    path = folder / PREPROCESSING_PATHS['static_information_file_path']
    datasets, attrs = hdf5_summary(path)
    print(f'\n{path.name}: {len(datasets)} datasets, {len(attrs)} attributes; ptdf {datasets["ptdf"]}')
    si = load_static_information(path)
    print(
        'load_static_information: ptdf', tuple(si.dynamic_information.ptdf.shape), 'slack', int(si.solver_config.slack)
    )
    mapping, attr_map = field_mapping(path)
    for name, target in mapping.items():
        print(f'  {name:40s} {datasets[name][0]!s:16s} -> {target}')
    print('attributes:')
    for name, target in attr_map.items():
        print(f'  {name:40s} {attrs[name]!s:16s} -> {target}')


if __name__ == '__main__':
    main()
