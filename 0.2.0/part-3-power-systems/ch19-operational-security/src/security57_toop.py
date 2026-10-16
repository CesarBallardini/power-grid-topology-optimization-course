"""Lab Part A of Chapter 19 with ToOp: the DC security metrics of IEEE 57 recomputed in NumPy from the full matrices.

Run it with ToOp's interpreter (its container or virtual environment) from this folder; ``metrics.py`` must be
next to it::

    python security57_toop.py /tmp/case57

It writes ToOp's IEEE 57 example (``case57_data_powsybl``), preprocesses it, obtains the full N-0 and N-1 flow
matrices with a pass-through aggregation function and checks every metric of the chapter against ToOp's own
``aggregate_to_metric``. ToOp's flows and limits are in MW; ``overload_weight`` is 2.0 on the lines of this case.
"""

import functools
import sys
from pathlib import Path

import jax
import metrics
import numpy as np
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case57_data_powsybl
from toop_engine_dc_solver.jax import default_topology, run_solver_symmetric
from toop_engine_dc_solver.jax.aggregate_results import aggregate_to_metric
from toop_engine_dc_solver.jax.types import ActionIndexComputations
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

jax.config.update('jax_enable_x64', True)

METRICS = (
    'overload_energy_n_0',
    'overload_energy_n_1',
    'exponential_overload_energy_n_1',
    'critical_branch_count_n_1',
    'n0_n1_delta',
    'overload_energy_limited_n_0',
    'overload_energy_limited_n_1',
)


def load(folder: Path):
    """Step 1: write the case, preprocess it and run the initial load flow; returns stats, static info, metrics."""
    case57_data_powsybl(folder)
    stats, si, _ = load_grid(
        DirFileSystem(str(folder)),
        pandapower=False,
        parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False),
    )
    si, unsplit = run_initial_loadflow(si)  # (overload_energy_n_0, overload_energy_n_1) of the unsplit grid
    return stats, si, unsplit


def passthrough(lf):
    """Aggregation function that returns the raw flow matrices. Defined once, never inside a loop."""
    return lf.n_0_matrix, lf.n_1_matrix


def matrices(si, topologies) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 2: N-0 (topology, timestep, branch), N-1 (topology, timestep, case, branch) and the success flags."""
    (n0, n1), ok = run_solver_symmetric(topologies, None, None, si.dynamic_information, si.solver_config, passthrough)
    return np.asarray(n0), np.asarray(n1), np.asarray(ok)


def limits(si) -> dict[str, np.ndarray]:
    """The arrays of ``branch_limits`` as NumPy, with the N-1 limit falling back to the N-0 one."""
    bl = si.dynamic_information.branch_limits
    n_1 = bl.max_mw_flow_n_1 if bl.max_mw_flow_n_1 is not None else bl.max_mw_flow
    return {
        'max_mw_flow': np.asarray(bl.max_mw_flow),
        'max_mw_flow_n_1': np.asarray(n_1),
        'overload_weight': np.asarray(bl.overload_weight) if bl.overload_weight is not None else None,
        'max_mw_flow_limited': np.asarray(bl.max_mw_flow_limited),
        'max_mw_flow_n_1_limited': np.asarray(bl.max_mw_flow_n_1_limited),
        'n0_n1_max_diff': np.asarray(bl.n0_n1_max_diff),
    }


def numpy_metrics(n0: np.ndarray, n1: np.ndarray, lim: dict[str, np.ndarray]) -> dict[str, float]:
    """Step 3 and 5: every metric of ``METRICS`` for one topology, from ``metrics.py`` and the chapter's formulas."""
    w = lim['overload_weight']
    delta = metrics.n0_n1_delta(n0, n1)  # (timesteps, branches)
    d = lim['n0_n1_max_diff']
    penalty = np.where(d < 0, 0.0, np.clip(delta - d, 0, None)).sum()
    return {
        'overload_energy_n_0': metrics.overload_energy_n1(n0[:, None, :], lim['max_mw_flow'], w),
        'overload_energy_n_1': metrics.overload_energy_n1(n1, lim['max_mw_flow_n_1'], w),
        'exponential_overload_energy_n_1': metrics.power_law_n1(n1, lim['max_mw_flow_n_1'], weight=w),
        'critical_branch_count_n_1': float(metrics.critical_branch_count_n1(n1, lim['max_mw_flow_n_1'])),
        'n0_n1_delta': float(penalty),
        'overload_energy_limited_n_0': metrics.overload_energy_n1(n0[:, None, :], lim['max_mw_flow_limited'], w),
        'overload_energy_limited_n_1': metrics.overload_energy_n1(n1, lim['max_mw_flow_n_1_limited'], w),
    }


def toop_metric(si, topologies, metric: str) -> np.ndarray:
    """Step 5: the same metric from ToOp, through ``aggregate_to_metric`` passed as the aggregation function."""
    di = si.dynamic_information
    fn = functools.partial(
        aggregate_to_metric,
        branch_limits=di.branch_limits,
        reassignment_distance=di.action_set.reassignment_distance,
        n_relevant_subs=si.n_sub_relevant,
        metric=metric,
    )
    values, _ = run_solver_symmetric(topologies, None, None, di, si.solver_config, fn)
    return np.asarray(values)


def with_actions(si, actions: dict[int, int], batch_size: int = 4) -> ActionIndexComputations:
    """Step 5: a batch of unsplit topologies with real action indices in the first slot of some of them."""
    base = default_topology(si.solver_config, batch_size=batch_size)
    act = base.action
    for topology, action in actions.items():
        act = act.at[topology, 0].set(action)
    return ActionIndexComputations(action=act, pad_mask=base.pad_mask)


def regimes(n1_unsplit: np.ndarray, rating: np.ndarray, limited: np.ndarray) -> dict[str, int]:
    """Step 4: how many branches fall in each regime of the double limit, from the unsplit worst flows."""
    loading = np.abs(n1_unsplit).max(axis=(0, 1)) / rating
    ratio = limited / rating
    return {
        'below 90 %, limit = 0.9 rating': int(((loading < 0.9) & np.isclose(ratio, 0.9)).sum()),
        'dead zone, limit = flow': int(((loading >= 0.9) & (loading <= 1.0) & np.isclose(ratio, loading)).sum()),
        'above 100 %, limit = rating': int(((loading > 1.0) & np.isclose(ratio, 1.0)).sum()),
    }


def main() -> None:
    """Steps 1 to 5 on IEEE 57, printing every number the solution quotes."""
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else 'case57')
    stats, si, unsplit = load(folder)
    print(
        f'stats: {stats.n_nodes} nodes, {stats.n_branches} branches, {stats.n_branch_outages} branch outages + '
        f'{stats.n_injection_outages} injection outages = {stats.n_nminus1_cases} N-1 cases, '
        f'{stats.n_relevant_subs} relevant substations, {stats.n_actions} actions, '
        f'double limits: {stats.has_double_limits}'
    )
    print('run_initial_loadflow (overload_energy_n_0, overload_energy_n_1) =', unsplit)
    lim = limits(si)
    w = lim['overload_weight']
    values, counts = np.unique(w, return_counts=True)
    print('overload weights:', {float(v): int(c) for v, c in zip(values, counts, strict=True)})
    print('N-1 limits / N-0 limits:', np.unique(np.round(lim['max_mw_flow_n_1'] / lim['max_mw_flow'], 6)))

    topologies = default_topology(si.solver_config, batch_size=4)
    n0, n1, ok = matrices(si, topologies)
    print(f'n_0 {n0.shape}, n_1 {n1.shape}, all cases successful: {ok.all()}')
    mine = numpy_metrics(n0[0], n1[0], lim)
    print(f'NumPy unsplit energies: N-0 {mine["overload_energy_n_0"]:.9f}, N-1 {mine["overload_energy_n_1"]:.9f}')
    print(
        f'  differences to ToOp: {abs(mine["overload_energy_n_0"] - unsplit[0]):.1e}, '
        f'{abs(mine["overload_energy_n_1"] - unsplit[1]):.1e}'
    )

    lim0 = metrics.double_limits(n0[0][:, None, :], lim['max_mw_flow'])
    lim1 = metrics.double_limits(n1[0], lim['max_mw_flow_n_1'])
    print(
        'double limits, max |mine - ToOp|: N-0',
        np.abs(lim0 - lim['max_mw_flow_limited']).max(),
        'N-1',
        np.abs(lim1 - lim['max_mw_flow_n_1_limited']).max(),
    )
    print('N-1 regimes:', regimes(n1[0], lim['max_mw_flow_n_1'], lim['max_mw_flow_n_1_limited']))
    print('n0_n1_max_diff factors: all negative (penalty disabled)?', bool((lim['n0_n1_max_diff'] < 0).all()))

    actions = {1: 1, 2: 5}
    topologies = with_actions(si, actions)
    n0, n1, ok = matrices(si, topologies)
    print(f'batch with actions {actions}: successful {ok.all(axis=1)}')
    for metric in METRICS:
        toop = toop_metric(si, topologies, metric)
        mine = np.array([numpy_metrics(n0[t], n1[t], lim)[metric] for t in range(len(toop))])
        print(f'  {metric:32s} ToOp {np.round(toop, 4)}  max |diff| {np.abs(toop - mine).max():.1e}')


if __name__ == '__main__':
    main()
