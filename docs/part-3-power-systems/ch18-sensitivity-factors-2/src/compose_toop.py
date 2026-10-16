"""Lab step 4 of Chapter 18 against ToOp: the NumPy composition of this folder versus ToOp's own references.

Run it with ToOp's interpreter (its container or virtual environment) from this folder::

    python compose_toop.py /tmp/case14

It writes ToOp's IEEE 14 example (``case14_pandapower``), loads it with ``load_grid`` and compares: the unsplit
flows of ``run_initial_loadflow`` with ``ieee14.load_case14().flows()``; and, for a bus split followed by an N-1
screen, the flows of ``compose.split`` + ``compose.n1_screen`` with ``run_solver`` of
``packages/dc_solver_pkg/tests/numpy_reference.py`` (found through ``$TOOP_REPO`` or the ``../ToOp`` sibling).
ToOp counts loads positive, so its injections and flows are the negatives of this folder's.
"""

import importlib.util
import os
import sys
from pathlib import Path

import jax
import numpy as np
from compose import moved_ends, n1_screen, split
from fsspec.implementations.dirfs import DirFileSystem
from ieee14 import DcGrid, load_case14
from toop_engine_dc_solver.example_grids import case14_pandapower
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

jax.config.update('jax_enable_x64', True)


def numpy_reference():
    """ToOp's NumPy reference solver, a test helper that is not part of the installed packages."""
    repo = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
    path = repo / 'packages' / 'dc_solver_pkg' / 'tests' / 'numpy_reference.py'
    spec = importlib.util.spec_from_file_location('numpy_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(folder: Path):
    """Write ToOp's IEEE 14 case into ``folder`` and preprocess it; returns static information and network data."""
    case14_pandapower(folder)
    _, si, nd = load_grid(
        DirFileSystem(str(folder)), pandapower=True, parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False)
    )
    return si, nd


def grid_from_toop(nd) -> DcGrid:
    """The ``DcGrid`` of ToOp's network data, with generation-positive injections and the course's branch names."""
    n = nd.n_original_nodes
    return DcGrid(
        frm=np.asarray(nd.from_nodes),
        to=np.asarray(nd.to_nodes),
        b=np.asarray(nd.susceptances),
        p=-np.asarray(nd.nodal_injection[0])[:n],
        names=load_case14().names,
        slack=int(nd.slack),
    )


def topo_vect(nd, station: int, bus_a: list[int]) -> np.ndarray:
    """ToOp's dense topology vector: one flag per branch end at the relevant nodes, True for bus B."""
    flags = []
    for node, branches in zip(nd.relevant_nodes, nd.branches_at_nodes, strict=True):
        flags += [bool(node == station and k not in bus_a) for k in branches]
    return np.array(flags)


def reference_n1(nd, station: int, bus_a: list[int], outages: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """N-1 flows (outages x branches) of the split grid from ToOp's reference, in this folder's sign convention."""
    ref = numpy_reference()
    n1, ok = ref.run_solver(
        branch_topo_vect=topo_vect(nd, station, bus_a),
        relevant_nodes=np.asarray(nd.relevant_nodes),
        slack=int(nd.slack),
        n_stat=int(nd.n_original_nodes),
        ptdf=np.asarray(nd.ptdf),
        susceptance=np.asarray(nd.susceptances),
        from_node=np.asarray(nd.from_nodes),
        to_node=np.asarray(nd.to_nodes),
        branches_at_nodes=[np.asarray(b) for b in nd.branches_at_nodes],
        branch_direction=[np.asarray(d) for d in nd.branch_direction],
        branches_to_outage=np.asarray(outages),
        nodal_injections=np.asarray(nd.nodal_injection[0]),
    )
    return -n1, ok


def composed_n1(grid: DcGrid, station: int, bus_a: list[int], outages: list[int]) -> np.ndarray:
    """The same screen with this folder's BSDF and LODF matrix."""
    s, _ = split(grid.ptdf(), grid.frm, grid.to, grid.b, grid.p, station, bus_a)
    frm, to = moved_ends(grid.frm, grid.to, station, bus_a, grid.n_bus)
    return n1_screen(s.h, s.f, frm, to, outages)


def main() -> None:
    """Compare unsplit flows and a split-plus-N-1 screen between this folder and ToOp."""
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'case14')
    si, nd = load(folder)
    si, metrics = run_initial_loadflow(si)
    grid = grid_from_toop(nd)
    unsplit = -np.asarray(si.dynamic_information.unsplit_flow)[0]
    print('unsplit metrics (overload_energy_n_0, overload_energy_n_1):', metrics, '(no ratings in case14)')
    print('unsplit flows: max |ToOp - ieee14.load_case14()| =', np.abs(unsplit - load_case14().flows()).max(), 'MW')
    print('               max |ToOp - grid built from NetworkData| =', np.abs(unsplit - grid.flows()).max(), 'MW')
    station, bus_a = 3, [grid.names.index('L2-4'), grid.names.index('L4-5')]  # bus 4, as in compose.py
    outages = [k for k in range(grid.n_branch) if nd.outaged_branch_mask[k]]
    ref, ok = reference_n1(nd, station, bus_a, outages)
    mine = composed_n1(grid, station, bus_a, outages)
    print(f'bus 4 split (L2-4 + L4-5 on bus A), {len(outages)} outages: reference success flags all True: {ok.all()}')
    print('  max |N-1 flows, this folder - numpy_reference.run_solver| =', np.abs(mine - ref).max(), 'MW')


if __name__ == '__main__':
    main()
