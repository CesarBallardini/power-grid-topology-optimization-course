"""Run the Octave files of the chapters' src folders through Octave's built-in test blocks, when Octave is installed.

Every ``.m`` file under ``docs/part-*/chNN-slug/src/`` ends with ``%!test`` blocks, run with Octave's ``test``
function. Without Octave the tests are skipped, not failed. Files that need MATPOWER skip their own blocks unless
MATPOWER is found (``$MATPOWER`` or ``%LOCALAPPDATA%\\matpower\\matpower<version>``) and put on Octave's path.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OCTAVE_FILES = sorted(ROOT.glob('docs/part-*/*/src/*.m'))


def find_octave() -> str | None:
    """Octave's command-line executable, on Windows, Linux and macOS.

    ``$OCTAVE`` wins; then ``octave-cli`` or ``octave`` on the PATH (the normal case on Linux and macOS); then the
    usual install folders that are not on the PATH: the Chocolatey/installer location on Windows, Homebrew and
    ``/opt`` on macOS and Linux.
    """
    env = os.environ.get('OCTAVE')
    if env:
        return env
    found = shutil.which('octave-cli') or shutil.which('octave')
    if found:
        return found
    if sys.platform == 'win32':
        roots = [Path(os.environ.get(v, '')) / 'GNU Octave' for v in ('ProgramFiles', 'ProgramFiles(x86)')]
        pattern = 'Octave-*/mingw64/bin/octave-cli.exe'
    else:
        roots = [Path('/opt'), Path('/opt/homebrew/bin'), Path('/usr/local/bin'), Path('/usr/bin')]
        pattern = '**/octave-cli'
    candidates = sorted(c for root in roots if root.is_dir() for c in root.glob(pattern))
    return str(candidates[-1]) if candidates else None


def find_matpower() -> Path | None:
    """The MATPOWER folder (the one holding ``install_matpower.m``), or ``None`` to let the MATPOWER tests skip.

    ``$MATPOWER`` wins; otherwise the newest ``matpower<version>`` under ``%LOCALAPPDATA%\\matpower`` on Windows
    and under ``~/.local/share/matpower`` or ``/opt/matpower`` elsewhere.
    """
    env = os.environ.get('MATPOWER')
    if env:
        return Path(env)
    if sys.platform == 'win32':
        roots = [Path(os.environ['LOCALAPPDATA']) / 'matpower'] if 'LOCALAPPDATA' in os.environ else []
    else:
        roots = [Path.home() / '.local/share/matpower', Path('/opt/matpower')]
    found = sorted(c for root in roots if root.is_dir() for c in root.glob('matpower*/install_matpower.m'))
    return found[-1].parent if found else None


OCTAVE = find_octave()
MATPOWER = find_matpower()


def test_every_octave_file_has_test_blocks():
    assert OCTAVE_FILES, 'no Octave sources found'
    for path in OCTAVE_FILES:
        assert '%!test' in path.read_text(encoding='utf-8'), f'{path} has no %!test block'


@pytest.mark.skipif(OCTAVE is None, reason='Octave is not installed')
@pytest.mark.parametrize('path', OCTAVE_FILES, ids=lambda p: p.stem)
def test_octave_file_passes_its_own_tests(path):
    setup = f"cd('{MATPOWER.as_posix()}'); install_matpower(1, 0, 0); " if MATPOWER else ''
    program = f"{setup}cd('{path.parent.as_posix()}'); [n, nmax] = test('{path.stem}'); exit(n ~= nmax)"
    cmd = [OCTAVE, '--no-gui', '--quiet', '--eval', program]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)  # noqa: S603
    assert result.returncode == 0, result.stdout + result.stderr
