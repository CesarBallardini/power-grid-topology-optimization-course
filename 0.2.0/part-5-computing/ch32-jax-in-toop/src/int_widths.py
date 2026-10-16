"""Platform integer widths: what NumPy and JAX call ``int`` on this machine (Chapter 32, Exercise 4).

Run it twice to see JAX's half of the answer: ``JAX_ENABLE_X64=1 python int_widths.py`` and
``JAX_ENABLE_X64=0 python int_widths.py``; ``main`` spawns the second run itself.
"""

import os
import platform
import subprocess
import sys

import numpy as np


def numpy_report() -> dict:
    """NumPy's default integer, its ``long`` and the largest random integer it will draw."""
    try:
        randint = int(np.random.default_rng(0).integers(2**32))
        randint_ok = True
    except ValueError:
        randint, randint_ok = None, False
    return {
        'numpy': np.__version__,
        'platform': f'{platform.system()} {platform.machine()}, Python {platform.python_version()}',
        'dtype(int)': np.dtype(int).name,
        "dtype('long')": np.dtype('long').name,
        'intp': np.dtype(np.intp).name,
        'randint(2**32) works': randint_ok,
        'randint value': randint,
    }


def jax_report() -> dict:
    """JAX's dtype for ``dtype=int`` and the int_max() sentinel under the current x64 setting."""
    import jax
    import jax.numpy as jnp

    x64 = bool(jax.config.read('jax_enable_x64'))
    full = jnp.full(3, 7, dtype=int)
    sentinel = int(jnp.iinfo(jnp.int64).max) if x64 else int(jnp.iinfo(jnp.int32).max)  # ToOp's int_max()
    wrapped = int(np.array(np.iinfo(np.int64).max).astype(np.int32))  # the 64-bit sentinel forced into int32
    return {
        'jax': jax.__version__,
        'x64': x64,
        'jnp.full(3, 7, dtype=int).dtype': full.dtype.name,
        'int_max()': sentinel,
        'int64 max stored as int32': wrapped,
    }


def jax_report_in_subprocess(enable_x64: bool) -> str:
    """Run ``jax_report`` in a fresh interpreter with ``JAX_ENABLE_X64`` set, since the flag is read at import."""
    env = dict(os.environ, JAX_ENABLE_X64='1' if enable_x64 else '0')
    code = 'import int_widths; print(int_widths.jax_report())'
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
    """NumPy's answer, then JAX's answer with and without x64."""
    for key, value in numpy_report().items():
        print(f'{key:22s} {value}')
    print('JAX_ENABLE_X64=1:', jax_report_in_subprocess(True))
    print('JAX_ENABLE_X64=0:', jax_report_in_subprocess(False))


if __name__ == '__main__':
    main()
