"""ToOp's separation set and switching distance for two stations (Chapter 13, Lab steps 1–4).

Runs with ToOp's interpreter only. The first station is the worked example's chain of four busbars and three
couplers with six line bays (the same station as ``reassignment_distance.py``), built here as a ``RuntimeBusGroup``
by hand. The second is voltage level S1VL2 of pypowsybl's four-substations node-breaker network, the double-busbar
level with one coupler, materialized with the same ToOp functions as ``notebooks/example_asset_topology_analysis``.
"""

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pypowsybl as pp
from reassignment_distance import ASSET_BUSBAR, ASSET_NAMES, N_BUSBARS, asset_table, candidates, reassignment_distance
from toop_engine_dc_solver.preprocess.helpers.switching_distance import per_station_switching_distance
from toop_engine_dc_solver.preprocess.preprocess_switching import (
    OptimalSeparationSetInfo,
    make_optimal_separation_set,
    prepare_for_separation_set,
)
from toop_engine_grid_helpers.powsybl.example_grids import create_default_network_masks
from toop_engine_grid_helpers.powsybl.powsybl_asset_topo import materialize_runtime_bus_groups_from_network_state
from toop_engine_grid_helpers.powsybl.powsybl_station_to_graph import get_node_breaker_master_asset_topology
from toop_engine_interfaces.asset_topology.assets_runtime import (
    RuntimeBranchAsset,
    RuntimeBusbar,
    RuntimeBusbarCoupler,
)
from toop_engine_interfaces.asset_topology.runtime_topology import RuntimeAssetConnection, RuntimeBusGroup
from toop_engine_interfaces.messages.preprocess.preprocess_commands import AreaSettings, CgmesImporterParameters

CHAIN_COUPLERS = [(0, 1), (1, 2), (2, 3)]


def chain_station() -> RuntimeBusGroup:
    """The worked example as a ToOp bus group: busbars b0..b3, closed breakers c01, c12, c23, six line bays."""
    busbars = [RuntimeBusbar(grid_model_id=f'b{i}', int_id=i) for i in range(N_BUSBARS)]
    couplers = [
        RuntimeBusbarCoupler(
            grid_model_id=f'c{a}{b}', coupler_type='BREAKER', busbar_from_id=a, busbar_to_id=b, open=False
        )
        for a, b in CHAIN_COUPLERS
    ]
    branches = [RuntimeAssetConnection(asset=RuntimeBranchAsset(grid_model_id=name)) for name in ASSET_NAMES]
    return RuntimeBusGroup(
        bus_group_id='chain',
        busbars=busbars,
        couplers=couplers,
        branch_connections=branches,
        injection_connections=[],
        branch_switching_table=asset_table(ASSET_BUSBAR, N_BUSBARS),
        injection_switching_table=np.zeros((N_BUSBARS, 0), dtype=bool),
    )


def four_substations_bus_groups(folder: Path) -> list[RuntimeBusGroup]:
    """The runtime bus groups of pypowsybl's four-substations network, one per voltage level.

    The network is saved as XIIDM into ``folder`` because ToOp's importer parameters name a grid file; every
    substation is marked relevant so that each voltage level becomes a bus group.
    """
    net = pp.network.create_four_substations_node_breaker_network()
    grid = folder / 'four_substations.xiidm'
    net.save(str(grid))
    masks = create_default_network_masks(net)
    masks = replace(
        masks,
        relevant_subs=np.ones(len(net.get_buses()), dtype=bool),
        busbar_for_nminus1=np.ones(len(net.get_busbar_sections()), dtype=bool),
    )
    parameters = CgmesImporterParameters(
        area_settings=AreaSettings(control_area=[''], view_area=[''], nminus1_area=[''], cutoff_voltage=1),
        data_folder=folder,
        grid_model_file=grid,
    )
    master = get_node_breaker_master_asset_topology(network=net, network_masks=masks, importer_parameters=parameters)
    return materialize_runtime_bus_groups_from_network_state(network=net, master_data=master)


def separation(group: RuntimeBusGroup) -> OptimalSeparationSetInfo:
    """Prepare a bus group and build its optimal separation set, as ToOp's preprocessing does."""
    branch_ids = [c.asset.grid_model_id for c in group.branch_connections]
    injection_ids = [c.asset.grid_model_id for c in group.injection_connections]
    simplified, _problems = prepare_for_separation_set(group, branch_ids, injection_ids)
    return make_optimal_separation_set(simplified)


def switching_distance(
    info: OptimalSeparationSetInfo, current_coupler_state: np.ndarray, target: np.ndarray
) -> tuple[int, bool, int, int]:
    """Best configuration, whether A and B were swapped, reassignment distance and coupler distance of a target."""
    best, invert, reassignment, coupler = per_station_switching_distance(
        jnp.asarray(target, dtype=bool),
        jnp.asarray(current_coupler_state, dtype=bool),
        jnp.asarray(info.separation_set),
        jnp.asarray(info.coupler_states),
    )
    return int(best), bool(invert), int(reassignment), int(coupler)


def describe_group(group: RuntimeBusGroup) -> str:
    """One line per busbar, coupler and asset of a bus group."""
    lines = [f'bus group {group.bus_group_id} (voltage level {group.voltage_level_id})']
    lines += [f'  busbar {b.grid_model_id} (int_id {b.int_id})' for b in group.busbars]
    lines += [
        f'  coupler {c.grid_model_id} {c.coupler_type} b{c.busbar_from_id}-b{c.busbar_to_id} open={c.open}'
        for c in group.couplers
    ]
    for row, busbar in zip(group.asset_switching_table, group.busbars, strict=True):
        assets = [a.grid_model_id for a, on in zip(group.assets, row, strict=True) if on]
        lines.append(f'  on {busbar.grid_model_id}: {assets}')
    return '\n'.join(lines)


def main(folder: Path) -> None:
    """Print the bus groups of the four-substations network, the separation sets and the switching distances."""
    groups = four_substations_bus_groups(folder)
    for group in groups:
        print(describe_group(group))
    s1vl2 = next(g for g in groups if g.voltage_level_id == 'S1VL2')
    info = separation(s1vl2)
    print(
        'S1VL2 separation set',
        info.separation_set.astype(int).tolist(),
        'coupler states',
        info.coupler_states.astype(int).tolist(),
    )
    print('S1VL2 busbar A', info.busbar_a, 'coupler distance', np.asarray(info.coupler_distance).tolist())

    chain = chain_station()
    info = separation(chain)
    print('chain: configurations', info.separation_set.shape[0], 'coupler states (True = open):')
    for state, group_a, row in zip(info.coupler_states, info.busbar_a, info.separation_set, strict=True):
        a = [n for n, on in zip(ASSET_NAMES, row[0], strict=True) if on]
        b = [n for n, on in zip(ASSET_NAMES, row[1], strict=True) if on]
        print(f'  open {state.astype(int).tolist()} busbars A {sorted(group_a)}: A = {a}, B = {b}')
    mine = candidates(CHAIN_COUPLERS, ASSET_BUSBAR)
    same = all(any((row == c).all() or (row[::-1] == c).all() for c in mine) for row in info.separation_set)
    print('chain: same candidates as coupler_splits:', same)
    closed = np.zeros(len(CHAIN_COUPLERS), dtype=bool)
    for label, target in (
        ('L3 L4 L5 on B', np.array([0, 0, 0, 1, 1, 1], dtype=bool)),
        ('L3 L4 on B (L5 moves)', np.array([0, 0, 0, 1, 1, 0], dtype=bool)),
        ('L0 L1 L2 on B (mirror)', np.array([1, 1, 1, 0, 0, 0], dtype=bool)),
    ):
        best, invert, reassignment, coupler = switching_distance(info, closed, target)
        numpy_best = int(reassignment_distance(mine, target).min())
        print(f'target {label}: best {best}, invert {invert}, reassignment {reassignment}, coupler {coupler}')
        print(f'  numpy reassignment_distance: {numpy_best}')


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()))
