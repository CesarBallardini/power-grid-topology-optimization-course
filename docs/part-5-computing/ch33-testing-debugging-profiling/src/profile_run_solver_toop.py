"""Time two ``run_solver`` calls on ToOp's case57 and trace them with the JAX profiler (Chapter 33, Exercise 6).

Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe profile_run_solver_toop.py [trace folder]

The first call traces, compiles and runs; the second only runs. The trace of both is written for
Perfetto (``perfetto_trace.json.gz``) and XProf (``*.xplane.pb``). Numbers are CPU numbers of the machine
that ran the script; py-spy is not part of this script (it is run from outside: ``py-spy record -o profile.svg
-- python profile_run_solver_toop.py``).
"""

import gzip
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import jax
from toop_engine_dc_solver.jax import default_topology, run_solver
from toop_engine_dc_solver.jax.inputs import load_static_information

jax.config.update('jax_enable_x64', True)

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
CASE57 = TOOP / 'data' / 'case57_data_powsybl' / 'static_information.hdf5'


def load(path=CASE57):
    """The two halves of ``static_information``: dynamic information (arrays) and solver config (static shapes)."""
    si = load_static_information(path)
    return si.dynamic_information, si.solver_config


def timed_calls(di, sc, batch_size=4, n_calls=2, trace_dir=None):
    """Seconds per ``run_solver`` call on the unsplit topology; every call is traced when ``trace_dir`` is set."""
    topologies = default_topology(sc, batch_size=batch_size)
    seconds = []
    if trace_dir is not None:
        jax.profiler.start_trace(str(trace_dir), create_perfetto_trace=True)
    try:
        for _ in range(n_calls):
            start = time.perf_counter()
            jax.block_until_ready(run_solver(topologies, None, None, di, sc))
            seconds.append(time.perf_counter() - start)
    finally:
        if trace_dir is not None:
            jax.profiler.stop_trace()
    return seconds


PATTERNS = {
    'whole run_solver calls': r'run_solver$',
    'tracing (trace_to_jaxpr_dynamic)': r'^trace_to_jaxpr_dynamic$',
    'XLA compilation (backend_compile)': r'^backend_compile$',
    'compiled executable calls (PjitFunction(iterate_symmetric_sequential))': r'^PjitFunction\(iterate_symmetric',
}


def trace_files(trace_dir):
    """The files the profiler wrote, relative to ``trace_dir``."""
    return sorted(str(p.relative_to(trace_dir)) for p in Path(trace_dir).rglob('*') if p.is_file())


def trace_totals(trace_dir, patterns=PATTERNS):
    """Seconds and counts of the outermost events whose name matches each pattern, from the Perfetto trace.

    Trace events nest (a function and the functions it calls are all recorded), so only the outermost match of
    each pattern is summed; the result is the inclusive time spent in that kind of event.
    """
    path = next(Path(trace_dir).rglob('perfetto_trace.json.gz'))
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        events = [e for e in json.load(handle)['traceEvents'] if e.get('ph') == 'X']
    events.sort(key=lambda e: (e['ts'], -e.get('dur', 0)))
    totals = {}
    for label, pattern in patterns.items():
        regex = re.compile(pattern)
        end, seconds, count = -1.0, 0.0, 0
        for event in events:
            if regex.search(event['name']) and event['ts'] >= end:
                end = event['ts'] + event.get('dur', 0)
                seconds += event.get('dur', 0) / 1e6
                count += 1
        totals[label] = (seconds, count)
    return totals


def main():
    """Load case57, time the calls, trace them and summarize the trace."""
    trace_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='jax_trace_'))
    di, sc = load()
    print(f'case57: PTDF {tuple(di.ptdf.shape)}, {int(di.ptdf.shape[0])} branches, batch size 4')
    first, second = timed_calls(di, sc)
    print(f'first run_solver call (trace + compile + run): {first:.3f} s')
    print(f'second call (compiled): {second:.4f} s, ratio {first / second:.0f}x')
    jax.clear_caches()
    first, second = timed_calls(di, sc, trace_dir=trace_dir)
    print(f'under jax.profiler.trace after clearing the caches: first {first:.3f} s, second {second:.4f} s')
    print('trace files:', trace_files(trace_dir))
    print('inclusive time of the outermost events in the trace:')
    for label, (seconds, count) in trace_totals(trace_dir).items():
        print(f'  {seconds:8.3f} s  x{count:<4d} {label}')
    print(f'open in Perfetto (https://ui.perfetto.dev): {trace_dir}')


if __name__ == '__main__':
    main()
