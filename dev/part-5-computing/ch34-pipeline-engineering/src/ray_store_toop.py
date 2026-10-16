"""Where Ray puts its object store, and what it says about /dev/shm (Chapter 34, Exercise 5).

Run it inside the ToOp container, once per ``--shm-size``::

    python ray_store_toop.py

Ray sizes its object store at 30 % of the memory it sees and, when ``/dev/shm`` is smaller than that, logs a
warning and puts the store in its temporary folder instead (disk-backed, slower). Nothing fails, which is why a
small ``/dev/shm`` does not show up as an error in a test run: pytest captures the warning with the rest of the
log. ``RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`` (as in ToOp's CI) stops Ray from packaging the working directory for
its workers when the driver was started by ``uv run``.
"""

import logging
import os
import shutil
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('RAY_ENABLE_UV_RUN_RUNTIME_ENV', '0')

import ray  # noqa: E402

SHM = Path('/dev/shm')  # noqa: S108 -- the shared-memory mount being measured, not a temporary file


class _Collect(logging.Handler):
    """Keep the messages of Ray's log records."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def object_store(put_mb=200):
    """Start a local Ray instance, put ``put_mb`` MB into its object store, read it back and shut down.

    Returns the ``/dev/shm`` size (``None`` where there is none), the object-store size Ray chose, whether Ray
    warned that the store left ``/dev/shm``, the time of the ``put`` and whether the data came back intact.
    """
    collect = _Collect()
    ray_logger = logging.getLogger('ray')
    ray_logger.addHandler(collect)
    try:
        ray.init(include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
        store_bytes = ray.available_resources()['object_store_memory']
        data = np.ones(put_mb * 2**20 // 8)
        start = time.perf_counter()
        ref = ray.put(data)
        put_seconds = time.perf_counter() - start
        intact = float(ray.get(ref).sum()) == data.size
    finally:
        ray.shutdown()
        ray_logger.removeHandler(collect)
    return {
        'shm_mb': round(shutil.disk_usage(SHM).total / 2**20) if SHM.is_dir() else None,
        'object_store_mb': round(store_bytes / 2**20),
        'fallback_warning': next((m for m in collect.messages if 'instead of /dev/shm' in m), None),
        'put_seconds': put_seconds,
        'intact': intact,
    }


def main():
    """Print the numbers quoted in the solution of Exercise 5."""
    result = object_store()
    print('ray', ray.__version__)
    print(f'{SHM} size MB:', result['shm_mb'])
    print('object store MB:', result['object_store_mb'])
    print(f'store moved out of {SHM}:', result['fallback_warning'] is not None)
    if result['fallback_warning']:
        print('  ' + result['fallback_warning'].split(' This will')[0])
    print(f'put 200 MB: {result["put_seconds"]:.2f} s, read back intact: {result["intact"]}')


if __name__ == '__main__':
    main()
