"""The first run of the pipeline as a script: what ``notebooks/example1_dc_loadflow_example.ipynb`` does (Chapter 0).

Run it inside the ToOp container (or with ToOp's virtual environment) from this folder::

    python case57_toop.py /tmp/case57

It writes the already imported IEEE 57 folder with ToOp's own generator, lists the files it contains before and
after the DC preprocessing, prints the node and branch counts of Exercise 5, and times the DC load flow twice to
separate compilation from execution (Lab step 2).
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case57_data_powsybl
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

jax.config.update('jax_enable_x64', True)

# Which stage of the pipeline writes each file of a processed grid folder (Lab step 3).
STAGE_OF = {
    'grid.xiidm': 'import (the grid model, pypowsybl format)',
    'loadflow_parameters.json': 'import (load-flow settings for the AC validation)',
    'masks': 'import (which lines, transformers, generators and substations the optimizer may use)',
    'initial_topology': 'import (the substations as busbars and couplers, for the AC stage)',
    'logs': 'import (start time of the run)',
    'static_information.hdf5': 'DC preprocessing (the arrays the GPU solver reads)',
    'static_information_stats.json': 'DC preprocessing (import statistics)',
    'action_set.json': 'DC preprocessing (every busbar split the optimizer may try)',
    'action_set_diffs.hdf5': 'DC preprocessing (the same actions as incidence changes)',
    'nminus1_definition.json': 'DC preprocessing (the contingency list, reused by the AC stage)',
}


def write_folder(folder: Path) -> list[str]:
    """Write the imported IEEE 57 folder with ToOp's generator into a new folder; return its entries, sorted."""
    if folder.exists() and any(folder.iterdir()):
        msg = f'{folder} is not empty: give a new folder, so that the listing shows what each stage writes'
        raise FileExistsError(msg)
    folder.mkdir(parents=True, exist_ok=True)
    case57_data_powsybl(folder)
    return entries(folder)


def entries(folder: Path) -> list[str]:
    """Top-level files and folders of a processed grid folder, sorted by name."""
    return sorted(p.name for p in folder.iterdir())


def load(folder: Path):
    """Preprocess the folder as the notebook does; return the static information and the network data."""
    _, si, network_data = load_grid(
        DirFileSystem(str(folder)),
        pandapower=False,
        parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False, fail_on_non_convergence=False),
    )
    return si, network_data


def counts(si) -> dict[str, int]:
    """Node, branch and timestep counts of the preprocessed grid (Exercise 5)."""
    dyn = si.dynamic_information
    return {'n_nodes': int(dyn.n_nodes), 'n_branches': int(dyn.n_branches), 'n_timesteps': int(dyn.n_timesteps)}


def node_type_counts(network_data) -> dict[str, int]:
    """How many solver nodes of each type (``BUS``, ``BUS_B``, ``PSTNode``, ...): the answer to Exercise 5."""
    types, n = np.unique(np.asarray(network_data.node_types).astype(str), return_counts=True)
    return dict(zip(types.tolist(), n.tolist(), strict=True))


def branch_type_counts(network_data) -> dict[str, int]:
    """How many branches of each type (``LINE``, ``TWO_WINDINGS_TRANSFORMER``)."""
    types, n = np.unique(np.asarray(network_data.branch_types).astype(str), return_counts=True)
    return dict(zip(types.tolist(), n.tolist(), strict=True))


def timed_loadflows(si) -> tuple[float, float]:
    """Seconds of two consecutive ``run_initial_loadflow`` calls: the first compiles, the second only runs."""
    out = []
    for _ in range(2):
        start = time.perf_counter()
        si, metrics = run_initial_loadflow(si)
        jax.block_until_ready(si.dynamic_information.unsplit_flow)
        out.append(time.perf_counter() - start)
    return out[0], out[1]


def main(folder: Path) -> None:
    """Print the device, the files per stage, the counts and the two timings."""
    print('devices', jax.devices())
    before = write_folder(folder)
    print('after the import stage :', before)
    si, network_data = load(folder)
    after = entries(folder)
    print('after DC preprocessing :', after)
    for name in after:
        print(f'  {name:32s} {STAGE_OF.get(name, "?")}')
    print(counts(si))
    print('nodes by type   ', node_type_counts(network_data))
    print('branches by type', branch_type_counts(network_data))
    print('relevant substations', np.asarray(network_data.relevant_nodes).tolist())
    first, second = timed_loadflows(si)
    print(f'load flow: first {first:.3f} s, second {second:.3f} s')
    print('unsplit flow shape', tuple(si.dynamic_information.unsplit_flow.shape))


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path('case57'))
