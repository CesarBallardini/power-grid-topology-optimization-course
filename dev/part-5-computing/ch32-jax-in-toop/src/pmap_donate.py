"""Four fake CPU devices, pmap with buffer donation, and what happens to the donated input (Chapter 32, Exercise 5).

The device count is an XLA flag read when JAX initializes, so this module sets it before importing JAX and
``run_in_subprocess`` runs the demonstration in a fresh interpreter (inside a notebook or a test, JAX is usually
already imported with a single device).
"""

import os
import subprocess
import sys

os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def demo() -> dict:
    """Map a doubling over a leading axis of 4 with pmap, donating the input, and look at the input afterwards."""
    doubled = jax.pmap(lambda x: 2.0 * x, donate_argnums=(0,))
    x = jnp.arange(12.0).reshape(4, 3)
    y = doubled(x)
    try:
        after = float(x.sum())
    except RuntimeError as exc:  # a deleted buffer on GPU/TPU
        after = type(exc).__name__ + ': ' + str(exc).splitlines()[0]
    jitted = jax.jit(lambda x: 2.0 * x, donate_argnums=(0,))
    z = jnp.arange(3.0)
    jitted(z)
    return {
        'jax': jax.__version__,
        'devices': [str(d) for d in jax.devices()],
        'output sum': float(y.sum()),
        'pmap input after donation': after,
        'pmap input is_deleted': x.is_deleted(),
        'jit input is_deleted': z.is_deleted(),
    }


def run_in_subprocess() -> str:
    """Run ``demo`` with the device-count flag set in a fresh interpreter and return its last output line."""
    env = dict(os.environ, XLA_FLAGS='--xla_force_host_platform_device_count=4')
    code = 'import pmap_donate; print(pmap_donate.demo())'
    out = subprocess.run(  # noqa: S603 - runs this interpreter on this module
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        cwd=os.path.dirname(__file__),
    )
    return out.stdout.strip().splitlines()[-1]


def main() -> None:
    """Print the report of this process (the flag was set at import time)."""
    for key, value in demo().items():
        print(f'{key:26s} {value}')


if __name__ == '__main__':
    main()
