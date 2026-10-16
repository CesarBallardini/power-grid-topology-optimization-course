"""Compile time, the recompilation trigger and the batch-size knobs of ToOp's DC solver (Chapter 32, Lab).

Run it with ToOp's interpreter (the container, or ``../ToOp/.venv``) from this folder::

    python solver_lab_toop.py [folder]

It writes ToOp's ``case57_data_powsybl`` grid into ``folder`` (default: a ``toop_course_ch32`` folder in the
system temporary directory) the first time, preprocesses it with ``load_grid`` and prints the tables of Lab
steps 2, 3 and 4. Every timing waits for the device with ``jax.block_until_ready``.
"""

import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import psutil
from fsspec.implementations.dirfs import DirFileSystem
from toop_engine_dc_solver.example_grids import case57_data_powsybl, case300_powsybl
from toop_engine_dc_solver.jax import default_topology, run_solver, run_solver_symmetric
from toop_engine_dc_solver.jax.topology_looper import iterate_symmetric_sequential
from toop_engine_dc_solver.preprocess.convert_to_jax import load_grid, run_initial_loadflow
from toop_engine_interfaces.folder_structure import PREPROCESSING_PATHS
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters

jax.config.update('jax_enable_x64', True)

WRITERS = {'case57': case57_data_powsybl, 'case300': case300_powsybl}


def grid_folder(name: str = 'case57', root: Path | None = None) -> Path:
    """Where the preprocessed grid lives: ``<root or system temp>/toop_course_ch32/<name>``."""
    return (root or Path(tempfile.gettempdir()) / 'toop_course_ch32') / name


def setup(folder: Path, name: str = 'case57'):
    """Write ToOp's example grid when missing, preprocess it and run the initial load flow.

    Returns:
        ToOp's ``StaticInformation``: ``.solver_config`` (static) and ``.dynamic_information`` (traced).
    """
    if not (folder / PREPROCESSING_PATHS['grid_file_path_powsybl']).exists():
        folder.mkdir(parents=True, exist_ok=True)
        WRITERS[name](folder)
    _, si, _ = load_grid(
        DirFileSystem(str(folder)),
        pandapower=False,
        parameters=PreprocessParameters(action_set_filter_bsdf_lodf=False, fail_on_non_convergence=False),
    )
    si, _ = run_initial_loadflow(si)
    return si


def timed(fn, *args, **kwargs) -> tuple:
    """Call ``fn``, wait for every array of the result, return ``(result, seconds)``."""
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args, **kwargs))
    return out, time.perf_counter() - start


def compiled_programs() -> int:
    """Number of compiled programs in the cache of ToOp's jitted core, ``iterate_symmetric_sequential``.

    ``jax.no_tracing()`` cannot wrap ``run_solver`` under ToOp's JAX 0.5.3: the Python glue around the core
    (padding the topologies, allocating the result storage) dispatches eager primitives, which that version counts
    as tracing. Counting the core's cache entries answers the same question exactly.
    """
    return iterate_symmetric_sequential._cache_size()  # noqa: SLF001 - JAX's own accessor for the compile cache


def traced(fn, *args, **kwargs) -> tuple:
    """Like ``timed`` but also reports whether the call compiled a new program for the solver core."""
    before = compiled_programs()
    out, seconds = timed(fn, *args, **kwargs)
    return out, seconds, compiled_programs() > before


def passthrough(lf):
    """Aggregation function returning the raw N-0 and N-1 matrices. Defined once, at module level."""
    return lf.n_0_matrix, lf.n_1_matrix


# --8<-- [start:compile]
def compile_vs_run(si, batch_size: int = 4) -> list[tuple[str, float, bool]]:
    """Lab 2: three ``run_solver`` calls, then ``run_solver_symmetric`` with a fixed and with a fresh function."""
    sc, di = si.solver_config, si.dynamic_information
    topologies = default_topology(sc, batch_size=batch_size)
    rows = []
    for k in range(3):
        _, seconds, retraced = traced(run_solver, topologies, None, None, di, sc)
        rows.append((f'run_solver, call {k + 1}', seconds, retraced))
    for k in range(3):
        _, seconds, retraced = traced(run_solver_symmetric, topologies, None, None, di, sc, passthrough)
        rows.append((f'run_solver_symmetric, passthrough defined once, call {k + 1}', seconds, retraced))
    for k in range(3):
        fresh = lambda lf: (lf.n_0_matrix, lf.n_1_matrix)  # noqa: E731 - the point is a new object per call
        _, seconds, retraced = traced(run_solver_symmetric, topologies, None, None, di, sc, fresh)
        rows.append((f'run_solver_symmetric, fresh lambda, call {k + 1}', seconds, retraced))
    return rows


# --8<-- [end:compile]


def pf_summary(result) -> tuple[tuple, float]:
    """Shape and largest value of ``result.n_1_results.pf_n_1_max``."""
    arr = np.asarray(result.n_1_results.pf_n_1_max)
    return arr.shape, float(np.nanmax(arr))


# --8<-- [start:trigger]
def trigger_study(si, batch_size: int = 4) -> list[dict]:
    """Lab 3: predict, then measure, what each change does to the compiled program."""
    sc, di = si.solver_config, si.dynamic_information
    topologies = default_topology(sc, batch_size=batch_size)
    rows = []

    def record(name, prediction, *args):
        out, seconds, retraced = traced(run_solver, *args)
        shape, largest = pf_summary(out)
        rows.append(
            {
                'change': name,
                'prediction': prediction,
                'retraced': retraced,
                'seconds': seconds,
                'shape': shape,
                'max': largest,
            }
        )
        return out

    record('baseline, first call', 'compiles', topologies, None, None, di, sc)
    record('baseline, second call', 'cached', topologies, None, None, di, sc)
    record('replace(sc), no field changed', 'recompiles', topologies, None, None, di, replace(sc))
    record(
        'replace(sc, number_most_affected=20)',
        'recompiles, 20 columns',
        topologies,
        None,
        None,
        di,
        replace(sc, number_most_affected=20),
    )
    sc_mutable = replace(sc)
    record('fresh copy of sc, warm-up', 'recompiles', topologies, None, None, di, sc_mutable)
    sc_mutable.number_most_affected = 20
    record(
        'same copy, sc.number_most_affected = 20 in place',
        'reuses incorrectly: old columns',
        topologies,
        None,
        None,
        di,
        sc_mutable,
    )
    di_inj = eqx.tree_at(lambda d: d.nodal_injections, di, di.nodal_injections * 1.1)
    record('nodal_injections * 1.1 only', 'reuses, flows unchanged', topologies, None, None, di_inj, sc)
    di_both = eqx.tree_at(lambda d: d.unsplit_flow, di_inj, di.unsplit_flow * 1.1)
    record('nodal_injections and unsplit_flow * 1.1', 'reuses, flows * 1.1', topologies, None, None, di_both, sc)
    di_half = eqx.tree_at(lambda d: d.branch_limits.max_mw_flow, di, di.branch_limits.max_mw_flow / 2)
    record('max_mw_flow / 2, same sc', 'reuses incorrectly: old limits baked in', topologies, None, None, di_half, sc)
    record('max_mw_flow / 2, replace(sc)', 'recompiles, new limits', topologies, None, None, di_half, replace(sc))
    record(
        'batch of 8 topologies',
        'reuses: 4 and 8 both pad to batch_size_bsdf',
        default_topology(sc, batch_size=8),
        None,
        None,
        di,
        sc,
    )
    return rows


# --8<-- [end:trigger]


def batch_sweep(si, n_topologies: int = 64, bsdf_sizes=(4, 8, 16, 32), injection_sizes=(1, 2, 4)) -> list[dict]:
    """Lab 4: compile time, topologies per second and RSS growth over a grid of the two batch sizes (CPU)."""
    rows = []
    process = psutil.Process()
    for b in bsdf_sizes:
        for i in injection_sizes:
            sc_k = replace(si.solver_config, batch_size_bsdf=b, batch_size_injection=i)
            topologies = default_topology(sc_k, batch_size=n_topologies)
            rss_before = process.memory_info().rss
            try:
                _, first = timed(run_solver, topologies, None, None, si.dynamic_information, sc_k)
                steady = [timed(run_solver, topologies, None, None, si.dynamic_information, sc_k)[1] for _ in range(3)]
            except Exception as exc:  # noqa: BLE001 - an out-of-memory or shape error is a data point here
                rows.append({'batch_size_bsdf': b, 'batch_size_injection': i, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            rows.append(
                {
                    'batch_size_bsdf': b,
                    'batch_size_injection': i,
                    'compile_s': first - statistics.median(steady),
                    'topologies_per_s': n_topologies / statistics.median(steady),
                    'rss_growth_mb': (process.memory_info().rss - rss_before) / 1e6,
                }
            )
    return rows


def main() -> None:
    """Lab 2 and 3 on case57; Lab 4 on case300 (first fifty substations relevant, as ToOp ships it)."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print('JAX', jax.__version__, 'on', jax.default_backend(), '-', jax.devices()[0].device_kind)
    si = setup(grid_folder('case57', root))
    di = si.dynamic_information
    print(
        f'case57: {di.n_nodes} nodes, {di.n_branches} branches, {si.solver_config.n_sub_relevant} relevant subs, '
        f'{di.n_nminus1_cases} N-1 cases, number_most_affected={si.solver_config.number_most_affected}'
    )
    print('\n== Lab 2: compile vs run')
    for name, seconds, retraced in compile_vs_run(si):
        print(f'{name:60s} {seconds * 1e3:9.1f} ms  {"traced" if retraced else "cached"}')
    print('\n== Lab 3: the recompilation trigger')
    for r in trigger_study(si):
        print(
            f'{r["change"]:52s} predicted: {r["prediction"]:38s} {"traced" if r["retraced"] else "cached":6s} '
            f'{r["seconds"] * 1e3:8.1f} ms  pf_n_1_max {r["shape"]} max {r["max"]:.4f}'
        )
    print('\n== Lab 4: batch sizes on case300 (CPU: topologies per second and RSS growth; no VRAM here)')
    si300 = setup(grid_folder('case300', root), 'case300')
    for r in batch_sweep(si300):
        if 'error' in r:
            print(f'bsdf {r["batch_size_bsdf"]:3d} inj {r["batch_size_injection"]:2d}  {r["error"]}')
        else:
            print(
                f'bsdf {r["batch_size_bsdf"]:3d} inj {r["batch_size_injection"]:2d}  compile {r["compile_s"]:6.2f} s  '
                f'{r["topologies_per_s"]:8.1f} topologies/s  RSS +{r["rss_growth_mb"]:7.1f} MB'
            )


if __name__ == '__main__':
    main()
