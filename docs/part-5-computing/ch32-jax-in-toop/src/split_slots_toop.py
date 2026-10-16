"""The price of unused split slots: solver time against the number of split slots (Chapter 32, Exercise 6).

The optimizer's ``max_num_splits`` is the second dimension of the action array that reaches ``run_solver``.
Here every slot holds the padding sentinel, so no substation is split, and the time still grows with the
number of slots: the BSDF stage works on every slot. Run it with ToOp's interpreter from this folder.
"""

import statistics
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from solver_lab_toop import grid_folder, setup, timed
from toop_engine_dc_solver.jax import default_topology, run_solver
from toop_engine_dc_solver.jax.types import ActionIndexComputations, int_max

jax.config.update('jax_enable_x64', True)


def unsplit_with_slots(sc, n_topologies: int, n_slots: int) -> ActionIndexComputations:
    """``n_topologies`` unsplit topologies whose action array has ``n_slots`` columns, all padding."""
    base = default_topology(sc, batch_size=n_topologies)
    action = jnp.full((n_topologies, n_slots), int_max(), dtype=base.action.dtype)
    return ActionIndexComputations(action=action, pad_mask=jnp.ones(n_topologies, dtype=bool))


def slot_sweep(si, slots=(1, 2, 4, 8), n_topologies: int = 64, repeats: int = 5) -> list[dict]:
    """Compile time and steady-state seconds per topology for each number of split slots."""
    sc, di = si.solver_config, si.dynamic_information
    rows = []
    for k in slots:
        topologies = unsplit_with_slots(sc, n_topologies, k)
        _, first = timed(run_solver, topologies, None, None, di, sc)
        steady = statistics.median(timed(run_solver, topologies, None, None, di, sc)[1] for _ in range(repeats))
        rows.append(
            {
                'slots': k,
                'compile_s': first - steady,
                'steady_s': steady,
                'us_per_topology': 1e6 * steady / n_topologies,
            }
        )
    return rows


def main() -> None:
    """Sweep 1 to 16 slots on case57 and on case300 (default_topology itself uses a single slot)."""
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    for name in ('case57', 'case300'):
        si = setup(grid_folder(name, root), name)
        sc, di = si.solver_config, si.dynamic_information
        print(
            f'\n{name}: {di.n_branches} branches, {di.n_nminus1_cases} N-1 cases, {sc.n_sub_relevant} relevant subs, '
            f'default_topology columns = {default_topology(sc, batch_size=1).action.shape[1]}, '
            f'batch_size_bsdf = {sc.batch_size_bsdf}'
        )
        rows = slot_sweep(si, slots=(1, 2, 4, 8, 16))
        print(f'{"slots":>5s} {"compile [s]":>12s} {"steady [ms]":>12s} {"us/topology":>12s} {"vs 1 slot":>10s}')
        for r in rows:
            print(
                f'{r["slots"]:5d} {r["compile_s"]:12.2f} {r["steady_s"] * 1e3:12.1f} {r["us_per_topology"]:12.1f} '
                f'{r["steady_s"] / rows[0]["steady_s"]:10.2f}'
            )


if __name__ == '__main__':
    main()
