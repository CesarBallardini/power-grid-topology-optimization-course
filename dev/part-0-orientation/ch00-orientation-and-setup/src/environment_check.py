"""Which device JAX uses, and how much of a first run is compilation (Chapter 0).

Run it in any environment that has JAX (the course environment, the ToOp container or the course JupyterHub)::

    python environment_check.py

It prints the JAX version and device list, says whether the work goes to a GPU or to the CPU, and times a small
jitted computation twice: the first call includes XLA compilation, the second one only runs the compiled code.
"""

import time
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

GPU_PLATFORMS = ('gpu', 'cuda', 'rocm')


def device_kind(devices: Sequence) -> str:
    """``'GPU'`` if any device's ``platform`` is a GPU backend, else ``'CPU'``.

    ``jax.devices()`` returns ``[CudaDevice(id=0)]`` on a working GPU setup and ``[CpuDevice(id=0)]`` otherwise;
    the platform attribute is what the banner of the ToOp container tests as well.
    """
    return 'GPU' if any(getattr(d, 'platform', '').lower() in GPU_PLATFORMS for d in devices) else 'CPU'


def report() -> dict[str, str]:
    """Version, default backend, device list and kind of device, as strings ready to paste into lab notes."""
    devices = jax.devices()
    return {
        'jax': jax.__version__,
        'backend': jax.default_backend(),
        'devices': repr(devices),
        'kind': device_kind(devices),
        'x64': str(jax.config.read('jax_enable_x64')),
    }


def time_twice(fn: Callable, *args) -> tuple[float, float]:
    """Wall-clock seconds of two consecutive calls of ``fn(*args)``, blocking until the result is ready.

    For a jitted function the first call compiles (XLA) and runs; the second one only runs. Their difference is the
    compilation cost that every fresh process (or restarted kernel) pays again.
    """
    out = []
    for _ in range(2):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        out.append(time.perf_counter() - start)
    return out[0], out[1]


@partial(jax.jit, static_argnums=0)
def solve_laplacian(n_nodes: int = 200):
    """Solve a small dense linear system, the kind of work the DC solver does, as a stand-in for a load flow."""
    a = jnp.eye(n_nodes) * 2.0 - jnp.eye(n_nodes, k=1) - jnp.eye(n_nodes, k=-1)
    return jnp.linalg.solve(a, jnp.ones(n_nodes))


def main() -> None:
    """Print the device report and the two timings."""
    for key, value in report().items():
        print(f'{key:8s} {value}')
    first, second = time_twice(solve_laplacian)
    print(f'first call  {first * 1e3:8.1f} ms  (compile + run)')
    print(f'second call {second * 1e3:8.1f} ms  (run only)')
    print(f'compilation share of the first call: {100 * (first - second) / first:.0f} %')


if __name__ == '__main__':
    main()
