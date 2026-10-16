"""ToOp's base-case DC flows (``unsplit_flow``) on its IEEE 57 example against pypowsybl (Chapter 16, Lab steps 1, 4).

Run it with ToOp's interpreter (its container or virtual environment)::

    python toop57.py /tmp/case57

It writes ToOp's extended IEEE 57 case in PowSyBl format into that folder, as
``notebooks/example1_dc_loadflow_example.ipynb`` does, loads it, runs the initial load flow and compares
``unsplit_flow`` with ``pypowsybl.loadflow.run_dc`` on the saved grid.
"""

import sys
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import pypowsybl as pyp
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import CGMES_DISTRIBUTED_SLACK, case57_data_powsybl
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

jax.config.update('jax_enable_x64', True)


def load(folder: Path):
    """Write the example case into ``folder`` and preprocess it; return static information and network data."""
    case57_data_powsybl(folder)
    _, si, network_data = load_grid(
        DirFileSystem(str(folder)),
        pandapower=False,
        parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False),
    )
    si, _ = run_initial_loadflow(si)
    return si, network_data


def toop_flows(si, network_data) -> pd.DataFrame:
    """ToOp's ``unsplit_flow`` of timestep 0 next to the backend's DC flows and AC-DC mismatch, by branch id (MW)."""
    ids = list(network_data.branch_ids)
    return pd.DataFrame(
        {
            'name': list(network_data.branch_names),
            'unsplit_flow': np.asarray(si.dynamic_information.unsplit_flow)[0],
            'basecase_dc': np.asarray(network_data.basecase_dc_branch_flows)[0],
            'ac_dc_mismatch': np.asarray(network_data.ac_dc_mismatch)[0],
        },
        index=ids,
    )


def powsybl_flows(folder: Path) -> pd.Series:
    """``p1`` of every line and transformer after ``run_dc`` with ToOp's parameters on the saved grid (MW)."""
    net = pyp.network.load(str(folder / PREPROCESSING_PATHS['grid_file_path_powsybl']))
    pyp.loadflow.run_dc(net, CGMES_DISTRIBUTED_SLACK)
    return pd.concat([net.get_lines()['p1'], net.get_2_windings_transformers()['p1']]).rename('powsybl_p1')


def compare(folder: Path) -> tuple[pd.DataFrame, object]:
    """ToOp flows next to pypowsybl's ``p1``; ``unsplit + p1`` is zero when the orientations are opposite."""
    si, network_data = load(folder)
    frame = toop_flows(si, network_data)
    frame['powsybl_p1'] = powsybl_flows(folder).reindex(frame.index)
    frame['unsplit + p1'] = frame['unsplit_flow'] + frame['powsybl_p1']
    frame['unsplit - basecase_dc'] = frame['unsplit_flow'] - frame['basecase_dc']
    return frame, si


def main() -> None:
    """Shape of ``unsplit_flow`` and the agreement with pypowsybl's DC flows."""
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'case57')
    pd.set_option('display.width', 140)
    frame, si = compare(folder)
    di = si.dynamic_information
    shape = np.asarray(di.unsplit_flow).shape
    print(f'unsplit_flow shape {shape} = (n_timesteps, n_branches): {di.n_timesteps} timestep(s), ', end='')
    print(f'{di.n_branches} branches, {di.n_nodes} nodes')
    print(f'nodal_injections shape {np.asarray(di.nodal_injections).shape}; ', end='')
    print(f'{frame.index.nunique()} distinct branch ids, {frame["name"].nunique()} distinct names')
    print(f'max |unsplit_flow - basecase_dc_branch_flows| = {frame["unsplit - basecase_dc"].abs().max():.2e} MW')
    print(f'max |unsplit_flow + p1| = {frame["unsplit + p1"].abs().max():.4f} MW (p1 has the opposite orientation)')
    mism = frame['ac_dc_mismatch'].abs()
    print(
        f'AC-DC mismatch stored for alpha: mean |m| {mism.mean():.2f} MW, max {mism.max():.2f} MW on {mism.idxmax()}'
    )
    print(frame.loc[frame['unsplit_flow'].abs().nlargest(5).index].round(3))


if __name__ == '__main__':
    main()
