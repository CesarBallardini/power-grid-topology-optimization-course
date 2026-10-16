"""The tyro command line of ToOp's DC optimizer and the Pydantic fields behind its flags (Chapter 34, Exercise 4).

Run with ToOp's interpreter from this folder::

    ../../../../../ToOp/.venv/Scripts/python.exe tyro_flags_toop.py

``python -m toop_engine_topology_optimizer.dc.main --help`` is run in a subprocess with ``PYTHONUTF8=1``: tyro
draws its help with box characters, which crash on a cp1252 Windows console (``UnicodeEncodeError`` in colorama).
"""

import os
import subprocess
import sys

from toop_engine_topology_optimizer.dc.main import CLIArgs
from toop_engine_topology_optimizer.interfaces.messages.commons import DescriptorDef
from toop_engine_topology_optimizer.interfaces.messages.dc_params import BatchedMEParameters, LoadflowSolverParameters

# flag -> (model, field): tyro turns a nested Pydantic model into dotted flags, snake_case becoming kebab-case
FLAGS = {
    '--fixed-files': (CLIArgs, 'fixed_files'),
    '--double-limits': (CLIArgs, 'double_limits'),
    '--ga-config.runtime-seconds': (BatchedMEParameters, 'runtime_seconds'),
    '--ga-config.random-seed': (BatchedMEParameters, 'random_seed'),
    '--ga-config.n-worst-contingencies': (BatchedMEParameters, 'n_worst_contingencies'),
    '--ga-config.me-descriptors.0.num-cells': (DescriptorDef, 'num_cells'),
    '--lf-config.max-num-splits': (LoadflowSolverParameters, 'max_num_splits'),
    '--lf-config.batch-size': (LoadflowSolverParameters, 'batch_size'),
}


def help_text():
    """The ``--help`` output of the DC optimizer CLI."""
    env = dict(os.environ, PYTHONUTF8='1')
    result = subprocess.run(  # noqa: S603
        [sys.executable, '-m', 'toop_engine_topology_optimizer.dc.main', '--help'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        env=env,
        check=True,
    )
    return result.stdout


def flag_table(text):
    """For every flag of ``FLAGS``: present in the help, and the field's declared default."""
    rows = {}
    for flag, (model, field) in FLAGS.items():
        info = model.model_fields[field]
        rows[flag] = {
            'present': flag in text,
            'model': model.__name__,
            'field': field,
            'default': info.get_default(call_default_factory=True),
        }
    return rows


def main():
    """Print the mapping and the boolean-flag convention."""
    text = help_text()
    print(f'help text: {len(text.splitlines())} lines')
    for flag, row in flag_table(text).items():
        status = 'ok ' if row['present'] else 'MISSING'
        print(f'  {flag:42s} {status} {row["model"]}.{row["field"]} (default {row["default"]!r})')
    booleans = [
        line.strip(' \u2502') for line in text.splitlines() if '--ga-config.no-' in line or '--lf-config.no-' in line
    ]
    print('boolean fields become a pair of flags, for example:', booleans[:2])


if __name__ == '__main__':
    main()
