"""Fill a shared-memory segment larger than /dev/shm and see how the process dies (Chapter 34, Exercise 5).

Run it inside the ToOp container, once per ``--shm-size``, with the size of the segment in MB::

    python shm_fill.py 200

Creating the segment always succeeds: ``ftruncate`` on tmpfs only records a length and reserves nothing. The
pages are allocated when they are first written, and when ``/dev/shm`` has none left the kernel cannot back the
page and sends ``SIGBUS`` to the writer. There is no exception and no "out of shared memory" message: the
process is simply killed, so the fill runs in a child process and this script reports how the child ended.
Standard library only; it runs anywhere ``multiprocessing.shared_memory`` exists (on Windows the segment is a
named file mapping and the fill succeeds).
"""

import shutil
import signal
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

SHM = Path('/dev/shm')  # noqa: S108 -- the shared-memory mount being measured, not a temporary file


def shm_bytes():
    """Size of the ``/dev/shm`` file system in bytes, or ``None`` where there is none (Windows, macOS)."""
    return shutil.disk_usage(SHM).total if SHM.is_dir() else None


def fill(mb):
    """Create a segment of ``mb`` MB, write a byte into every position, then release it."""
    segment = shared_memory.SharedMemory(create=True, size=mb * 2**20)
    try:
        segment.buf[:] = b'\x01' * segment.size
    finally:
        segment.close()
        segment.unlink()


def fill_in_child(mb):
    """Run :func:`fill` in a child interpreter and return its return code (negative: killed by that signal)."""
    result = subprocess.run([sys.executable, __file__, '--child', str(mb)], capture_output=True, check=False)  # noqa: S603
    return result.returncode


def describe(returncode):
    """``exit 0``, ``exit 1`` or the name of the signal that killed the child (``SIGBUS``)."""
    if returncode >= 0:
        return f'exit {returncode}'
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f'signal {-returncode}'


def main():
    """Print the numbers quoted in the solution of Exercise 5."""
    if len(sys.argv) == 3 and sys.argv[1] == '--child':
        fill(int(sys.argv[2]))
        return
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    size = shm_bytes()
    print(f'{SHM} size MB:', 'none' if size is None else round(size / 2**20))
    returncode = fill_in_child(mb)
    print(f'fill of {mb} MB: return code {returncode} ({describe(returncode)})')


if __name__ == '__main__':
    main()
