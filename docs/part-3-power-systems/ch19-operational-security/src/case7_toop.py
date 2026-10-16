"""Hand the 7-bus case of Chapter 19 to ToOp: DC N-1 of a dispatch, then every single-substation split.

Run it inside the ToOp container (or with ToOp's virtual environment) from this folder; it needs ``case7.py``
next to it::

    python case7_toop.py /tmp/case7

It writes the pandapower grid and the masks ToOp expects into that folder, the way
``toop_engine_dc_solver.example_grids.case14_pandapower`` does, and prints the answers to Exercises 7-10.
"""

import sys
from pathlib import Path

import case7
import jax
import jax.numpy as jnp
import numpy as np
import pandapower as pp
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import random_topology_info
from toop_engine_dc_solver.jax import default_topology, run_solver_symmetric
from toop_engine_dc_solver.jax.types import ActionIndexComputations
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow
from toop_engine_grid_helpers.powsybl.powsybl_helpers import save_lf_params_to_fs
from toop_engine_interfaces.folder_structure import NETWORK_MASK_NAMES, PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

jax.config.update('jax_enable_x64', True)

RELEVANT = (3, 4, 5, 6)  # substations ToOp may split; it keeps only those with four or more branches


def write_case(folder: Path, pg: np.ndarray, limit: np.ndarray = case7.LIMIT, relevant=RELEVANT) -> None:
    """Write ``grid.json`` and the masks for the dispatch ``pg`` (one value per generator of ``case7.GENS``)."""
    net = case7.pandapower_net(pg, limit)
    pp.rundcpp(net)
    grid = folder / PREPROCESSING_PATHS['grid_file_path_pandapower']
    grid.parent.mkdir(parents=True, exist_ok=True)
    pp.to_json(net, grid)
    masks = folder / PREPROCESSING_PATHS['masks_path']
    masks.mkdir(parents=True, exist_ok=True)
    relevant_mask = np.zeros(len(net.bus), dtype=bool)
    relevant_mask[[bus - 1 for bus in relevant]] = True
    np.save(masks / NETWORK_MASK_NAMES['relevant_subs'], relevant_mask)
    all_lines = np.ones(len(net.line), dtype=bool)
    np.save(masks / NETWORK_MASK_NAMES['line_for_nminus1'], all_lines)
    np.save(masks / NETWORK_MASK_NAMES['line_for_reward'], all_lines)
    no_trafo = np.zeros(len(net.trafo), dtype=bool)
    np.save(masks / NETWORK_MASK_NAMES['trafo_for_nminus1'], no_trafo)
    np.save(masks / NETWORK_MASK_NAMES['trafo_for_reward'], no_trafo)
    # Line outages only, as in the SciPy code; with True here ToOp adds one N-1 row per generator outage
    np.save(masks / NETWORK_MASK_NAMES['generator_for_nminus1'], np.zeros(len(net.gen), dtype=bool))
    random_topology_info(folder)
    save_lf_params_to_fs({}, DirFileSystem(folder), Path(PREPROCESSING_PATHS['loadflow_parameters_file_path']))


def load(folder: Path):
    """Preprocess the folder; return ToOp's static information, the network data and the unsplit metrics."""
    _, si, network_data = load_grid(
        DirFileSystem(str(folder)), pandapower=True, parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False)
    )
    si, metrics = run_initial_loadflow(si)  # (overload_energy_n_0, overload_energy_n_1)
    return si, network_data, metrics


def passthrough(lf):
    """Aggregation function that returns the raw flow matrices. Defined once, never inside a loop."""
    return lf.n_0_matrix, lf.n_1_matrix


def matrices(si, topologies) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """N-0 flows (topology, branch), N-1 flows (topology, outage, branch) and the solver flags, timestep 0."""
    (n0, n1), ok = run_solver_symmetric(topologies, None, None, si.dynamic_information, si.solver_config, passthrough)
    return np.asarray(n0)[:, 0], np.asarray(n1)[:, 0], np.asarray(ok)


def unsplit(si) -> ActionIndexComputations:
    """One topology without any split."""
    return default_topology(si.solver_config, batch_size=1)


def single_splits(si) -> ActionIndexComputations:
    """One topology per action of the action set: each applies a single action to a single substation."""
    base = unsplit(si)
    n_actions = int(np.sum(np.asarray(si.dynamic_information.action_set.n_actions_per_sub)))
    act = jnp.full((n_actions, base.action.shape[1]), jnp.iinfo(jnp.int32).max, dtype=base.action.dtype)
    act = act.at[:, 0].set(jnp.arange(n_actions))
    return ActionIndexComputations(action=act, pad_mask=jnp.ones(n_actions, dtype=bool))


def describe(si, network_data, action: int) -> str:
    """Which substation an action splits and which branches it moves to a busbar of their own."""
    action_set = si.dynamic_information.action_set
    sub = int(np.asarray(action_set.substation_correspondence)[action])
    names = list(network_data.branch_names)
    at_sub = [names[i] for i in network_data.branches_at_nodes[sub]]
    flags = np.asarray(action_set.branch_actions)[action, : len(at_sub)]
    moved = [name for name, flag in zip(at_sub, flags, strict=True) if flag]
    bus = f'bus{int(np.asarray(network_data.relevant_nodes)[sub]) + 1}'
    return f'{bus} unsplit' if not moved else f'{bus} split, {"+".join(moved)} on their own busbar'


def report(folder: Path, pg: np.ndarray, limit: np.ndarray = case7.LIMIT) -> None:
    """Run the unsplit N-1 and every single split for one dispatch and print what the exercises ask for."""
    write_case(folder, pg, limit)
    si, network_data, metrics = load(folder)
    names = list(network_data.branch_names)
    lim = np.asarray(si.dynamic_information.branch_limits.max_mw_flow)
    outages = [names[i] for i in np.flatnonzero(np.asarray(network_data.outaged_branch_mask))]
    print(f'dispatch {np.round(pg, 1)}: unsplit (overload_energy_n_0, overload_energy_n_1) = {np.round(metrics, 3)}')

    (n0,), (n1,), _ = matrices(si, unsplit(si))
    n1 = n1[: len(outages)]  # the rows are the branch outages, followed by injection outages if any
    print('N-0 loading %:', dict(zip(names, np.round(100 * np.abs(n0) / lim, 1), strict=True)))
    for o, row in zip(outages, np.abs(n1), strict=True):
        k = int(row.argmax())
        print(f'  outage {o:5s}: worst {names[k]:5s} at {100 * row[k] / lim[k]:5.1f} %')

    per_sub = np.asarray(si.dynamic_information.action_set.n_actions_per_sub)
    subs = [f'bus{int(i) + 1}' for i in np.asarray(network_data.relevant_nodes)]
    print('actions per relevant substation (one of each is the unsplit one):', dict(zip(subs, per_sub, strict=True)))
    n0_all, n1_all, ok_all = matrices(si, single_splits(si))
    n1_all = n1_all[:, : len(outages)]
    worst = (np.abs(n1_all) / lim).max(axis=(1, 2)) * 100
    print('per action: largest N-0 flow change against unsplit, worst post-contingency loading')
    for a, w in enumerate(worst):
        k_o, k_b = np.unravel_index(np.abs(n1_all[a] / lim).argmax(), n1_all[a].shape)
        print(
            f'  action {a:2d}: {describe(si, network_data, a):45s} dN-0 {np.abs(n0_all[a] - n0).max():6.1f} MW, '
            f'worst {w:6.1f} % ({names[k_b]} after {outages[k_o]} out), solver ok={bool(ok_all[a].all())}'
        )
    print(f'splits that clear every N-1 overload: {int((worst <= 100 + 1e-9).sum())} of {len(worst)}')


def main() -> None:
    """Exercises 7-9 with the OPF dispatch, Exercise 10 with the SCOPF dispatch and a tighter line 3-6."""
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'case7')
    print('== Exercises 7-9: the OPF dispatch of the worked example ==')
    report(folder / 'opf', case7.dispatch().x)
    print('\n== Exercise 10: the SCOPF dispatch with the limit of L3-6 lowered to 100 MW ==')
    lim = case7.LIMIT.copy()
    lim[case7.NAMES.index('L3-6')] = 100
    report(folder / 'scopf', case7.dispatch(range(case7.N_LINE)).x, lim)


if __name__ == '__main__':
    main()
