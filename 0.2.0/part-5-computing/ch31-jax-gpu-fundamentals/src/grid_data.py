"""pandapower test grids as plain arrays, and the NumPy N-1 screening the JAX version is checked against (Lab 1)."""

import warnings

import numpy as np
import pandapower.networks as pn
from pandapower.converter.pypower import to_ppc


def branch_data(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """From bus, to bus, series reactance and net injection of a pandapower test case.

    The case is converted to MATPOWER's ppc form, whose bus numbers are consecutive from zero, so the arrays
    can index a PTDF directly. Lines and transformers are both branches here; out-of-service branches are
    dropped.

    Args:
        case: Name of a function of ``pandapower.networks`` (``'case118'``, ``'case2869pegase'``, ...).

    Returns:
        ``(from_bus, to_bus, x, p)``: integer bus indices, reactances in per unit, and the net injection per bus
        in MW (generation minus load; the slack absorbs the imbalance, which the PTDF ignores).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # pandapower's own test cases ship with deprecated columns
        net = getattr(pn, case)()
        ppc = to_ppc(net, init='flat')
    branch, bus, gen = ppc['branch'].real, ppc['bus'].real, ppc['gen'].real
    position = {int(number): k for k, number in enumerate(bus[:, 0])}
    in_service = branch[:, 10] == 1
    from_bus = np.array([position[int(b)] for b in branch[in_service, 0]])
    to_bus = np.array([position[int(b)] for b in branch[in_service, 1]])
    x = branch[in_service, 3]
    p = -bus[:, 2].copy()  # PD column: load
    on = gen[:, 7] == 1
    np.add.at(p, [position[int(b)] for b in gen[on, 0]], gen[on, 1])  # PG at GEN_BUS
    return from_bus, to_bus, x, p


def numpy_ptdf(from_bus: np.ndarray, to_bus: np.ndarray, x: np.ndarray, n_bus: int) -> np.ndarray:
    """PTDF with bus 0 as slack (zero column), the Chapter 17 construction in NumPy."""
    m = len(from_bus)
    a = np.zeros((m, n_bus))
    a[np.arange(m), from_bus] = 1.0
    a[np.arange(m), to_bus] = -1.0
    bf = a / x[:, None]
    bbus = a.T @ bf
    ptdf = np.zeros((m, n_bus))
    ptdf[:, 1:] = np.linalg.solve(bbus[1:, 1:], bf[:, 1:].T).T
    return ptdf


def numpy_screen(ptdf: np.ndarray, from_bus: np.ndarray, to_bus: np.ndarray, p: np.ndarray) -> tuple:
    """N-1 flows of every single branch outage with a Python loop; bridges give a zero row and a False flag."""
    f0 = ptdf @ p
    m = len(from_bus)
    flows, ok = np.zeros((m, m)), np.ones(m, dtype=bool)
    for k in range(m):
        col = ptdf[:, from_bus[k]] - ptdf[:, to_bus[k]]
        denom = 1.0 - col[k]
        if abs(denom) < 1e-11:
            ok[k] = False
            continue
        lodf = col / denom
        lodf[k] = -1.0
        flows[k] = f0 + lodf * f0[k]
    return flows, ok


def main() -> None:
    """Sizes of the lab's grids and the number of bridges each one has."""
    for case in ('case14', 'case118', 'case300', 'case1354pegase', 'case2869pegase'):
        from_bus, to_bus, x, p = branch_data(case)
        n_bus = len(p)
        if n_bus <= 300:
            _, ok = numpy_screen(numpy_ptdf(from_bus, to_bus, x, n_bus), from_bus, to_bus, p)
            bridges = f'{int((~ok).sum())} bridges'
        else:
            bridges = 'bridges not counted (the NumPy loop is slow here)'
        print(f'{case:15s} {n_bus:5d} buses {len(x):5d} branches  {bridges}')


if __name__ == '__main__':
    main()
