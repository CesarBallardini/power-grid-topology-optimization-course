"""A Hydra multirun of ToOp's benchmark on a copy of the node-breaker grid, and its assessment (Chapter 34, Ex. 3).

Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe hydra_sweep_toop.py [work folder]

The grid file is copied into the work folder, so ToOp's own ``data/`` is never written. The benchmark is run in a
subprocess (``--run``) that first applies ``windows_safe_log_dir`` from ``tensorboard_log_toop``: at ``ba1874e``
every job of the sweep dies on Windows after preprocessing with ``WinError 123`` because of the colons in the
TensorBoard directory name. ``ga_config.runtime_seconds`` and ``ga_config.n_worst_contingencies`` are swept (the
latter is a real ``BatchedMEParameters`` field; the page's ``ga_config.split_subs`` is not).
"""

import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TOOP = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
BENCHMARK = TOOP / 'toop-engine-benchmark'
GRID_XIIDM = TOOP / 'data' / 'grid_node_breaker' / 'grid.xiidm'


def overrides(grid_file, output_dir, runtimes=(4, 8), n_worst=(1, 2)):
    """The Hydra command-line overrides of the sweep."""
    return [
        f'grid.grid_file={Path(grid_file).as_posix()}',
        f'grid.output_dir={Path(output_dir).as_posix()}',
        'ga_config.runtime_seconds=' + ','.join(str(r) for r in runtimes),
        'ga_config.n_worst_contingencies=' + ','.join(str(k) for k in n_worst),
    ]


def run_sweep(work_dir, **kwargs):
    """Copy the grid, run the multirun in a subprocess and return the multirun folder and the exit code.

    At ``ba1874e`` every job ends with ``ConfigKeyError: Missing key critical_va_diff_degree``: the benchmark
    builds an ``ac_validation_cfg`` with ``n_processes`` and ``k_best_topos`` only, and ``save_ac_metrics_summary``
    reads two more keys. The DC stage of every job completes before that, so the exit code is 1 and the tree holds
    the Hydra folders and the DC results, but no ``benchmark_summary.json``.
    """
    work_dir = Path(work_dir)
    grid_dir = work_dir / 'data' / 'gnb'
    grid_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(GRID_XIIDM, grid_dir / 'grid.xiidm')
    output_dir = work_dir / 'results'
    env = dict(os.environ, PYTHONUTF8='1', PWD=str(TOOP))
    result = subprocess.run(  # noqa: S603
        [sys.executable, __file__, '--run', *overrides(grid_dir / 'grid.xiidm', output_dir, **kwargs)],
        cwd=str(TOOP),
        env=env,
        check=False,
    )
    return output_dir / 'multirun', result.returncode


def assess(multirun_dir, report_path):
    """Run ``assess_benchmarks`` on the sweep; return its JSON report, or None when no job wrote a summary."""
    env = dict(os.environ, PYTHONUTF8='1', PWD=str(TOOP))
    report_path = Path(report_path)
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(BENCHMARK / 'assess_benchmarks.py'),
            f'root={Path(multirun_dir).as_posix()}',
            f'save={Path(report_path).as_posix()}',
            'print=false',
        ],
        cwd=str(TOOP),
        env=env,
        check=True,
        capture_output=True,
    )
    return json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else None


def tree(root, skip=('stats', 'masks', 'tensorboard', 'sld')):
    """The directory tree of the sweep as indented lines, without the bulky leaves."""
    root = Path(root)
    lines = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part in skip for part in rel.parts[:-1]) or rel.parts[-1] in skip:
            if rel.parts[-1] in skip and path.is_dir():
                lines.append('  ' * (len(rel.parts) - 1) + rel.parts[-1] + '/ ...')
            continue
        lines.append('  ' * (len(rel.parts) - 1) + rel.parts[-1] + ('/' if path.is_dir() else ''))
    return lines


def job_summaries(multirun_dir):
    """Per Hydra job: its overrides and the best DC fitness of the ``stats/res.json`` its DC stage wrote."""
    rows = []
    for job in sorted(p for p in Path(multirun_dir).rglob('.hydra') if p.is_dir()):
        job_dir = job.parent
        if not any(job_dir.glob('*/stats/res.json')):
            continue
        res = json.loads(next(job_dir.glob('*/stats/res.json')).read_text(encoding='utf-8'))
        job_overrides = (job / 'overrides.yaml').read_text(encoding='utf-8').split()
        rows.append(
            {
                'job': job_dir.name,
                'overrides': [o for o in job_overrides if o.startswith('ga_config')],
                'max_fitness': res['max_fitness'],
            }
        )
    return rows


def _run_in_process(argv):
    """Entry point of the subprocess: install the Windows shim, then run the Hydra app as ``__main__``.

    Hydra resolves ``config_path="configs"`` relative to the script only when the decorated function lives in
    ``__main__``; imported as a module it would look for a Python package called ``configs``.
    """
    import tensorboard_log_toop

    tensorboard_log_toop.windows_safe_log_dir()
    script = BENCHMARK / 'benchmark_toop.py'
    sys.argv = [str(script), '--multirun', *argv]
    runpy.run_path(str(script), run_name='__main__')


def main():
    """Run the sweep, print the tree and the assessment."""
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        _run_in_process(sys.argv[2:])
        return
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='toop_sweep_'))
    multirun, code = run_sweep(work_dir)
    print(f'benchmark exit code {code}')
    print('\n'.join(tree(work_dir)))
    for row in job_summaries(multirun):
        print(row)
    logs = sorted(multirun.rglob('benchmark_toop.log'))
    if logs:
        tail = [
            line for line in logs[0].read_text(encoding='utf-8').splitlines() if 'Error' in line or 'error' in line
        ]
        print('first job log, error lines:', tail[-3:])
    report = assess(multirun, work_dir / 'aggregate_report.json')
    print('assessment report:', 'none written (no benchmark_summary.json)' if report is None else list(report)[:12])


if __name__ == '__main__':
    main()
