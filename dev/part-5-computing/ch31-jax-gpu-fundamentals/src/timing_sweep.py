"""Compile time against steady-state time of the N-1 screening, per grid size and precision (Chapter 31, Lab 3).

Run as ``python timing_sweep.py`` for the four grids of the lab (a few minutes on a laptop CPU). Set
``JAX_PLATFORMS=cpu`` for the CPU, or leave JAX to pick the GPU when it has one (a JAX with the CUDA plugin). Two
more modes::

    python timing_sweep.py repeat 3   # one warm-up sweep, then the median of three sweeps
    python timing_sweep.py agree      # the same screening on the GPU and on the CPU, compared entry by entry
"""

import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from grid_data import branch_data
from n1_screen import ptdf_matrix, screen

CASES = ('case118', 'case300', 'case1354pegase', 'case2869pegase')


def time_screen(ptdf, from_bus, to_bus, p, repeats: int = 10) -> tuple[float, float]:
    """Seconds of the first call (compilation included) and the median of the next ``repeats`` calls.

    Every call waits for its result with ``block_until_ready``; without that, JAX's asynchronous dispatch
    returns before the device has finished and the timer measures nothing. The cache is cleared first so that
    the first call always compiles, even when the same shapes were seen earlier in the process.
    """
    jax.clear_caches()
    outages = jnp.arange(ptdf.shape[0])
    start = time.perf_counter()
    jax.block_until_ready(screen(ptdf, from_bus, to_bus, p, outages))
    first = time.perf_counter() - start
    later = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(screen(ptdf, from_bus, to_bus, p, outages))
        later.append(time.perf_counter() - start)
    return first, statistics.median(later)


def sweep(cases: tuple[str, ...] = CASES, repeats: int = 10) -> list[dict]:
    """One row per grid and precision: branch count, output size, first-call and steady-state seconds.

    The PTDF is built once in float64 and cast, so the float32 rows measure the screening itself in float32.
    """
    rows = []
    for case in cases:
        from_bus, to_bus, x, p = (jnp.array(v) for v in branch_data(case))
        ptdf64 = ptdf_matrix(from_bus, to_bus, x, len(p))
        for dtype in (jnp.float64, jnp.float32):
            ptdf, inj = ptdf64.astype(dtype), p.astype(dtype)
            first, steady = time_screen(ptdf, from_bus, to_bus, inj, repeats)
            n_branch = ptdf.shape[0]
            rows.append(
                {
                    'case': case,
                    'n_branch': n_branch,
                    'dtype': jnp.dtype(dtype).name,
                    'output_mb': n_branch * n_branch * jnp.dtype(dtype).itemsize / 1e6,
                    'first_s': first,
                    'steady_s': steady,
                }
            )
    return rows


def repeated_sweep(cases: tuple[str, ...] = CASES, repeats: int = 10, runs: int = 3) -> list[dict]:
    """``sweep`` once as a warm-up (backend start-up, file caches, GPU clocks), then ``runs`` times; the medians.

    Every run clears the compilation cache per row, so each ``first_s`` is a full compilation again.
    """
    sweep(cases, repeats)
    results = [sweep(cases, repeats) for _ in range(runs)]
    return [
        {**rows[0], **{k: statistics.median(r[k] for r in rows) for k in ('first_s', 'steady_s')}}
        for rows in zip(*results, strict=True)
    ]


def agreement(case: str) -> dict:
    """The screening of ``case`` on the default device (the GPU, if any) against the CPU, and float32 against float64.

    Each device builds its own PTDF and screens every outage. Flows are compared on the outages that both runs
    flag as valid; ``missed_bridges`` counts the bridges the float32 run does not flag (its denominator is
    rounding noise instead of zero, far above the ``1e-11`` threshold), whose flows look valid and are not.
    """
    data = branch_data(case)
    out = {'case': case, 'devices': (str(jax.devices()[0]), str(jax.devices('cpu')[0]))}
    runs = {}
    for name, device, dtype in (
        ('default64', jax.devices()[0], jnp.float64),
        ('cpu64', jax.devices('cpu')[0], jnp.float64),
        ('default32', jax.devices()[0], jnp.float32),
    ):
        with jax.default_device(device):
            from_bus, to_bus, x, p = (jax.device_put(jnp.array(v), device) for v in data)
            ptdf = ptdf_matrix(from_bus, to_bus, x, len(p)).astype(dtype)
            flows, ok = screen(ptdf, from_bus, to_bus, p.astype(dtype), jnp.arange(ptdf.shape[0]))
            runs[name] = (np.asarray(flows, dtype=np.float64), np.asarray(ok))
    for label, (a, b) in {
        'device_vs_cpu_float64': ('default64', 'cpu64'),
        'float32_vs_float64': ('default32', 'default64'),
    }.items():
        (fa, oka), (fb, okb) = runs[a], runs[b]
        both = oka & okb
        diff = np.abs(fa[both] - fb[both]).max()
        out[label] = {
            'max_abs_mw': float(diff),
            'max_rel': float(diff / np.abs(fb[both]).max()),
            'flags_equal': bool((oka == okb).all()),
            'missed_bridges': int((oka & ~okb).sum()),
            'missed_max_flow_mw': float(np.abs(fa[oka & ~okb]).max()) if (oka & ~okb).any() else None,
        }
    return out


def main() -> None:
    """Print the table the log-log plot is drawn from (time against number of branches, one line per dtype)."""
    mode = sys.argv[1] if len(sys.argv) > 1 else 'once'
    print('backend:', jax.default_backend(), '-', jax.devices()[0].device_kind, '- JAX', jax.__version__)
    if mode == 'agree':
        for case in CASES:
            print(agreement(case))
        return
    rows = repeated_sweep(runs=int(sys.argv[2])) if mode == 'repeat' else sweep()
    print(f'{"case":15s} {"branches":>8s} {"dtype":>8s} {"output MB":>10s} {"first [s]":>10s} {"steady [ms]":>11s}')
    for r in rows:
        print(
            f'{r["case"]:15s} {r["n_branch"]:8d} {r["dtype"]:>8s} {r["output_mb"]:10.1f} '
            f'{r["first_s"]:10.3f} {r["steady_s"] * 1e3:11.3f}'
        )


if __name__ == '__main__':
    main()
