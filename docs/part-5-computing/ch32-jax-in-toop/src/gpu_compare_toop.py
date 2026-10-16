"""ToOp's DC solver on CPU and on an NVIDIA GPU: compile time, batch sizes, memory and agreement (Chapter 32, Lab).

Run it with ToOp's interpreter in an image that has the CUDA plugin of ToOp's JAX (``jax[cuda12]``), from this
folder. The backend is chosen when JAX starts, by ``JAX_PLATFORMS`` (``cuda,cpu`` for the GPU, since ToOp's
preprocessing also asks for the CPU device; ``cpu`` for the CPU), with ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` so
that the memory figures are real use, not JAX's 75 % preallocation::

    python gpu_compare_toop.py env
    python gpu_compare_toop.py compile ROOT [REPEATS]
    python gpu_compare_toop.py sweep ROOT [float32] [bsdf=4,16,...] [inj=1,4,...]
    python gpu_compare_toop.py agree ROOT [case300|case57] [N_TOPOLOGIES]

``ROOT`` holds the preprocessed grids (``solver_lab_toop.grid_folder``). ``compile`` and ``sweep`` start a fresh
Python process per measurement: a compilation happens once per process, and JAX's peak-memory counter cannot be
reset, so a new process is the only clean slate. ``agree`` runs the same case300 topologies on both devices in one
process and prints how far apart the results are.
"""

import importlib.metadata
import json
import platform
import shutil
import statistics
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from jax.experimental import disable_x64
from solver_lab_toop import compile_vs_run, grid_folder, passthrough, setup, timed
from toop_engine_dc_solver.jax import default_topology, random_topology, run_solver, run_solver_symmetric

SMI_FIELDS = (
    'memory.used',
    'clocks.gr',
    'power.draw',
    'temperature.gpu',
    'utilization.gpu',
    'clocks_event_reasons.active',
)
BSDF_SIZES = (4, 16, 64, 256, 1024, 2048, 4096)
INJECTION_SIZES = (1, 4, 16, 64)
N_TOPOLOGIES = 1024


def nvidia_smi(fields: tuple[str, ...]) -> list[str] | None:
    """One ``nvidia-smi --query-gpu`` reading of the first GPU, or ``None`` when there is no ``nvidia-smi``."""
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return None
    out = subprocess.run(  # noqa: S603 - a fixed command, found on the PATH by shutil.which
        [smi, f'--query-gpu={",".join(fields)}', '--format=csv,noheader,nounits'],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        return None
    return [v.strip() for v in out.stdout.splitlines()[0].split(',')]


def environment() -> dict:
    """Versions of Python, JAX, jaxlib and the CUDA plugin, the devices JAX sees, and the GPU's state."""

    def version(dist: str) -> str | None:
        try:
            return importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            return None

    gpu_fields = (
        'name',
        'memory.total',
        'driver_version',
        'pstate',
        'clocks.gr',
        'clocks.max.gr',
        'clocks.mem',
        'power.draw',
        'power.limit',
        'temperature.gpu',
        'clocks_event_reasons.active',
    )
    reading = nvidia_smi(gpu_fields)
    return {
        'python': platform.python_version(),
        'jax': jax.__version__,
        'jaxlib': version('jaxlib'),
        'jax_cuda12_plugin': version('jax-cuda12-plugin'),
        'cuda_runtime': version('nvidia-cuda-runtime-cu12'),
        'cudnn': version('nvidia-cudnn-cu12'),
        'backend': jax.default_backend(),
        'devices': [str(d) for d in jax.devices()],
        'device_kind': jax.devices()[0].device_kind,
        'gpu': dict(zip(gpu_fields, reading, strict=True)) if reading else None,
    }


class SmiSampler:
    """Polls ``nvidia-smi`` in a thread while a measurement runs: memory, clock, power, temperature, throttling.

    ``nvidia-smi`` reports the memory of the whole card (the CUDA context included), so the peak is compared with
    the reading taken just before the measurement. The clock is the median over the samples in which the GPU was
    busy, and ``throttle`` ORs the clock-event bit masks of those samples (``0x4``: software power cap, ``0x20``
    and ``0x40``: thermal slowdown). With no ``nvidia-smi`` every figure is ``None``.
    """

    def __init__(self, interval: float = 0.05):
        """Sample every ``interval`` seconds once the context is entered."""
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self.available = nvidia_smi(SMI_FIELDS) is not None
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _parse(reading: list[str]) -> list[float]:
        *values, reasons = reading
        numbers = [float(v) if v.replace('.', '', 1).isdigit() else np.nan for v in values]
        return [*numbers, float(int(reasons, 16)) if reasons.startswith('0x') else np.nan]

    def _run(self) -> None:
        while not self._stop.is_set():
            reading = nvidia_smi(SMI_FIELDS)
            if reading:
                self.samples.append(self._parse(reading))
            self._stop.wait(self.interval)

    def __enter__(self):
        """Take the baseline reading and start sampling."""
        self.baseline = self._parse(nvidia_smi(SMI_FIELDS)) if self.available else None
        if self.available:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        """Stop sampling."""
        self._stop.set()
        if self.available:
            self._thread.join()

    def summary(self) -> dict:
        """Peak and baseline memory [MiB], busy clock [MHz], peak power [W], peak temperature [C], throttle mask."""
        keys = ('smi_peak_mib', 'smi_baseline_mib', 'clock_mhz', 'power_w', 'temp_c', 'throttle')
        if not self.samples:
            return dict.fromkeys(keys)
        s = np.array(self.samples)
        busy = s[s[:, 4] > 0] if np.any(s[:, 4] > 0) else s
        mask = 0
        for v in busy[:, 5]:
            mask |= int(v) if np.isfinite(v) else 0
        values = (s[:, 0].max(), self.baseline[0], np.median(busy[:, 1]), s[:, 2].max(), s[:, 3].max())
        return {**dict(zip(keys, (float(v) for v in values), strict=False)), 'throttle': hex(mask)}


class RssSampler:
    """Peak resident set of this process [bytes] while the context is open, sampled every ``interval`` seconds."""

    def __init__(self, interval: float = 0.02):
        """Sample every ``interval`` seconds once the context is entered."""
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self._stop.wait(self.interval)

    def __enter__(self):
        """Start sampling."""
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        """Stop sampling."""
        self._stop.set()
        self._thread.join()


def to_float32(tree):
    """A copy of a ToOp pytree in 32 bits: float64 to float32, int64 to int32 (the int64 sentinel to int32's).

    ToOp's preprocessing switches ``jax_enable_x64`` on when it is imported, so ``JAX_ENABLE_X64=0`` alone does
    not give a float32 run. Convert the static information and the topologies with this function, *then* switch
    x64 off (``jax.config.update('jax_enable_x64', False)``), so that every array the solver creates is 32-bit too.
    """

    def convert(x):
        a = np.asarray(x)
        if a.dtype == np.float64:
            return jnp.asarray(a.astype(np.float32))
        if a.dtype == np.int64:
            sentinel = a == np.iinfo(np.int64).max
            return jnp.asarray(np.where(sentinel, np.iinfo(np.int32).max, a).astype(np.int32))
        return jnp.asarray(a)

    return jax.tree.map(convert, tree)


def device_memory_mb() -> tuple[float | None, float | None]:
    """JAX's peak of device memory in use since the process started, and its allocator's limit [MB].

    Only the GPU backend keeps these statistics; on the CPU both are ``None``.
    """
    stats = jax.devices()[0].memory_stats() or {}
    if 'peak_bytes_in_use' not in stats:
        return None, None
    return stats['peak_bytes_in_use'] / 1e6, stats['bytes_limit'] / 1e6


# --8<-- [start:point]
def measure_point(si, bsdf: int, injection: int, n_topologies: int, repeats: int = 5, float32: bool = False) -> dict:
    """One point of the Lab 4 sweep on the current backend, in a process that has done nothing else.

    The first call compiles and runs; one more call warms up; the median of ``repeats`` calls is the steady state.
    ``compile_s`` is the first call minus that median. Memory: JAX's peak bytes in use on the GPU, the
    ``nvidia-smi`` peak of the card, and on the CPU the peak growth of the process's resident set, sampled every
    20 ms (``solver_lab_toop.batch_sweep`` reads the resident set only after the calls, when XLA has already
    freed its temporaries).
    """
    sc = replace(si.solver_config, batch_size_bsdf=bsdf, batch_size_injection=injection)
    di = si.dynamic_information
    topologies = default_topology(sc, batch_size=n_topologies)
    if float32:
        di, topologies = to_float32(di), to_float32(topologies)
        jax.config.update('jax_enable_x64', False)
    row = {'bsdf': bsdf, 'injection': injection, 'n_topologies': n_topologies, 'float32': float32}
    rss_before = psutil.Process().memory_info().rss
    try:
        with SmiSampler() as smi, RssSampler() as rss:
            _, first = timed(run_solver, topologies, None, None, di, sc)
            timed(run_solver, topologies, None, None, di, sc)
            steady = statistics.median(timed(run_solver, topologies, None, None, di, sc)[1] for _ in range(repeats))
    except Exception as exc:  # noqa: BLE001 - out of memory is the data point the sweep is looking for
        message = str(exc).splitlines()[0][:160]
        return {**row, 'error': f'{type(exc).__name__}: {message}'}
    return {
        **row,
        'compile_s': first - steady,
        'steady_s': steady,
        'topologies_per_s': n_topologies / steady,
        **dict(zip(('peak_device_mb', 'device_limit_mb'), device_memory_mb(), strict=True)),
        'rss_peak_growth_mb': (rss.peak - rss_before) / 1e6,
        **smi.summary(),
    }


# --8<-- [end:point]


def in_fresh_process(*args: str) -> list[dict]:
    """Run this script with ``args`` in a new interpreter and collect the JSON lines it prints."""
    out = subprocess.run(  # noqa: S603 - runs this interpreter on this module
        [sys.executable, str(Path(__file__).resolve()), *args], capture_output=True, text=True, check=False
    )
    rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith('{')]
    if not rows:
        tail = (out.stderr.strip().splitlines() or ['no output'])[-1]
        rows = [{'error': f'exit {out.returncode}: {tail[:160]}'}]
    return rows


def sweep(root: str, float32: bool = False, bsdf_sizes=BSDF_SIZES, injection_sizes=INJECTION_SIZES) -> list[dict]:
    """Lab 4: every ``(batch_size_bsdf, batch_size_injection)`` point in its own process, on case300.

    ``n_topologies`` is ``N_TOPOLOGIES`` or ``bsdf`` when larger, so that every point runs at least one full
    batch. Once a BSDF size runs out of memory for every injection size, larger ones are not tried.
    """
    rows = []
    for b in bsdf_sizes:
        line = []
        for i in injection_sizes:
            args = ['_point', root, str(b), str(i), str(max(N_TOPOLOGIES, b))] + (['float32'] if float32 else [])
            line.extend(in_fresh_process(*args))
        rows.extend(line)
        if all('error' in r for r in line):
            break
    return rows


def compile_runs(root: str, repeats: int = 3) -> dict[str, list[float]]:
    """Lab 2 in ``repeats`` fresh processes after one discarded warm-up process: the times of every call."""
    in_fresh_process('_compile', root)  # warm-up: file caches, GPU clocks
    times: dict[str, list[float]] = {}
    for _ in range(repeats):
        for r in in_fresh_process('_compile', root):
            times.setdefault(r['call'], []).append(r['seconds'])
    return times


# --8<-- [start:agree]
def solve_on(device, si, topologies) -> dict:
    """``pf_n_1_max``, the N-0 and N-1 flows and the success flags of ``topologies``, computed on ``device``."""
    with jax.default_device(device):
        di, topo = jax.device_put(si.dynamic_information, device), jax.device_put(topologies, device)
        sparse = run_solver(topo, None, None, di, si.solver_config)
        (n_0, n_1), success = run_solver_symmetric(topo, None, None, di, si.solver_config, passthrough)
        out = {'pf_n_1_max': sparse.n_1_results.pf_n_1_max, 'n_0': n_0, 'n_1': n_1, 'success': success}
        return {k: np.asarray(v) for k, v in jax.block_until_ready(out).items()}


def differences(a: dict, b: dict) -> dict:
    """Largest absolute and relative differences of two ``solve_on`` results, on the entries both trust.

    An outage that islands the grid has ``success`` False and a meaningless flow row (a division by a
    denominator that is zero up to rounding). The flows and ``pf_n_1_max`` are therefore compared on the
    topologies whose outages all succeeded, the ones ToOp's ``success`` accepts. ``n_0_max_abs_flagged`` and
    ``n_1_max_abs_flagged`` are the differences on the other topologies (N-1: their valid outages only); a split
    that makes an outage island the grid can make the whole solve ill-conditioned.
    ``missed_failures`` counts the outages ``b`` flags as islanding and ``a`` does not, and ``missed_max_flow`` is
    the largest flow ``a`` reports for them: numbers that look valid and are not. A topology whose N-0 flows are
    not finite (a split this precision cannot solve) is left out of every comparison and counted in
    ``unsolved_topologies``.
    """
    solved = np.isfinite(a['n_0']).all(axis=(1, 2)) & np.isfinite(b['n_0']).all(axis=(1, 2))
    ok = a['success'] & b['success'] & solved[:, None]
    whole = ok.all(axis=1)
    pairs = {
        'n_0': (a['n_0'][whole], b['n_0'][whole]),
        'n_1': (a['n_1'][whole, 0], b['n_1'][whole, 0]),
        'pf_n_1_max': (a['pf_n_1_max'][whole], b['pf_n_1_max'][whole]),
    }
    out = {
        'topologies': len(ok),
        'unsolved_topologies': (
            int((~np.isfinite(a['n_0']).all(axis=(1, 2))).sum()),
            int((~np.isfinite(b['n_0']).all(axis=(1, 2))).sum()),
        ),
        'failed_outages': (int((~a['success']).sum()), int((~b['success']).sum())),
        'success_flags_equal': bool((a['success'] == b['success']).all()),
        'topologies_compared_pf': int(whole.sum()),
    }
    flagged = solved & ~whole
    rows = ok & flagged[:, None]
    out['n_0_max_abs_flagged'] = float(np.abs(a['n_0'] - b['n_0'])[flagged].max()) if flagged.any() else None
    out['n_1_max_abs_flagged'] = float(np.abs(a['n_1'] - b['n_1'])[:, 0][rows].max()) if rows.any() else None
    missed = a['success'] & ~b['success']
    out['missed_failures'] = int(missed.sum())
    out['missed_max_flow'] = float(np.max(np.abs(a['n_1'][:, 0][missed]))) if missed.any() else None
    for name, (x, y) in pairs.items():
        out[f'{name}_max_abs'] = float(np.max(np.abs(x - y)))
        out[f'{name}_max_rel'] = float(np.max(np.abs(x - y)) / np.max(np.abs(y)))
        out[f'{name}_scale'] = float(np.max(np.abs(y)))
    return out


def agreement(si, n_topologies: int = 64, seed: int = 0) -> dict:
    """CPU against GPU in float64, and GPU float32 against GPU float64, on the same random case300 topologies.

    The topologies split up to two of the relevant substations each (``random_topology``), so the BSDF, the
    LODF update and the islanding checks are all exercised, not only the unsplit grid.
    """
    topologies = random_topology(
        jax.random.PRNGKey(seed), si.dynamic_information.action_set, limit_n_subs=2, batch_size=n_topologies
    )
    gpu, cpu = jax.devices()[0], jax.devices('cpu')[0]
    r64_gpu, r64_cpu = solve_on(gpu, si, topologies), solve_on(cpu, si, topologies)
    si32, topo32 = to_float32(si), to_float32(topologies)
    with disable_x64():
        r32_gpu = solve_on(gpu, si32, topo32)
    return {
        'devices': (str(gpu), str(cpu)),
        'cpu_vs_gpu_float64': differences(r64_gpu, r64_cpu),
        'gpu_float32_vs_float64': differences(r32_gpu, r64_gpu),
    }


# --8<-- [end:agree]


def main() -> None:
    """Dispatch the subcommands of the module docstring and print their tables."""
    command, args = sys.argv[1], sys.argv[2:]
    if command == 'env':
        print(json.dumps(environment(), indent=1))
    elif command == '_compile':
        si = setup(grid_folder('case57', Path(args[0])))
        for name, seconds, retraced in compile_vs_run(si):
            print(json.dumps({'call': name, 'seconds': seconds, 'traced': retraced}))
    elif command == 'compile':
        print('backend:', jax.default_backend(), '-', jax.devices()[0].device_kind)
        repeats = int(args[1]) if len(args) > 1 else 3
        for call, seconds in compile_runs(args[0], repeats).items():
            ms = ', '.join(f'{s * 1e3:.1f}' for s in seconds)
            print(f'{call:60s} median {statistics.median(seconds) * 1e3:8.1f} ms  ({ms})')
    elif command == '_point':
        si = setup(grid_folder('case300', Path(args[0])), 'case300')
        b, i, n = (int(v) for v in args[1:4])
        print(json.dumps(measure_point(si, b, i, n, float32='float32' in args[4:])))
    elif command == 'sweep':
        print('backend:', jax.default_backend(), '-', jax.devices()[0].device_kind)
        options = dict(a.split('=') for a in args[1:] if '=' in a)
        sizes = {k: tuple(int(v) for v in options[k].split(',')) for k in ('bsdf', 'inj') if k in options}
        grid = {'bsdf_sizes': sizes.get('bsdf', BSDF_SIZES), 'injection_sizes': sizes.get('inj', INJECTION_SIZES)}
        for r in sweep(args[0], float32='float32' in args[1:], **grid):
            head = f'bsdf {r.get("bsdf", 0):5d} inj {r.get("injection", 0):3d} n {r.get("n_topologies", 0):5d}'
            if 'error' in r:
                print(head, r['error'])
                continue
            gpu = (
                f'  peak {r["peak_device_mb"]:7.0f} MB  smi {r["smi_peak_mib"]:5.0f}/{r["smi_baseline_mib"]:4.0f} MiB'
                f'  {r["clock_mhz"]:5.0f} MHz {r["power_w"]:5.1f} W {r["temp_c"]:3.0f} C {r["throttle"]}'
                if r['peak_device_mb'] is not None and r['smi_peak_mib'] is not None
                else f'  peak RSS +{r["rss_peak_growth_mb"]:7.1f} MB'
            )
            print(f'{head}  compile {r["compile_s"]:6.2f} s  {r["topologies_per_s"]:9.1f} topologies/s{gpu}')
    elif command == 'agree':
        case = args[1] if len(args) > 1 else 'case300'
        si = setup(grid_folder(case, Path(args[0])), case)
        n = int(args[2]) if len(args) > 2 else 64
        print(case, n, 'topologies:', json.dumps(agreement(si, n_topologies=n), indent=1))
    else:
        raise SystemExit(f'unknown command {command!r}; see the module docstring')


if __name__ == '__main__':
    main()
