"""State estimation on pandapower's case14: noisy measurements, a bad datum, a topology error (Chapter 22, Exercise 4).

Measurements follow pandapower's conventions: bus p and q are the net consumption at the bus (``res_bus``, load
positive) without the shunt, which the estimator models as a network element; branch p and q are taken at the from
(hv) side. Standard deviations are 1 % for voltages and 2 % of the value (at least 0.1 MW or Mvar) for powers.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandapower as pp
import pandapower.estimation as est
import pandapower.networks as pn
import pandas as pd

SIGMA_V, SIGMA_P, FLOOR = 0.01, 0.02, 0.1


def solved_case14() -> pp.pandapowerNet:
    """case14 with its AC power flow solved."""
    net = pn.case14()
    pp.runpp(net)
    return net


def _add(
    net: pp.pandapowerNet, kind: str, element_type: str, value: float, sigma: float, element: int, side=None
) -> None:
    pp.create_measurement(net, kind, element_type, value, sigma, element=element, side=side)


def add_measurements(net: pp.pandapowerNet, rng: np.random.Generator | None) -> None:
    """Voltage magnitude at every bus, p and q at every bus, line (from side) and transformer (hv side).

    With ``rng`` the values carry Gaussian noise of the stated standard deviations; with ``None`` they are exact.
    """
    net.measurement = net.measurement.iloc[0:0]
    noise = (lambda s: rng.normal(0.0, s)) if rng is not None else (lambda s: 0.0)
    shunt_p = net.res_shunt.groupby(net.shunt.bus).p_mw.sum()
    shunt_q = net.res_shunt.groupby(net.shunt.bus).q_mvar.sum()
    for b in net.bus.index:
        v = net.res_bus.vm_pu[b]
        _add(net, 'v', 'bus', v + noise(SIGMA_V * v), SIGMA_V * v, b)
        p = net.res_bus.p_mw[b] - shunt_p.get(b, 0.0)
        q = net.res_bus.q_mvar[b] - shunt_q.get(b, 0.0)
        for kind, value in (('p', p), ('q', q)):
            s = max(SIGMA_P * abs(value), FLOOR)
            _add(net, kind, 'bus', value + noise(s), s, b)
    for table, element_type, side, cols in (
        (net.res_line, 'line', 'from', ('p_from_mw', 'q_from_mvar')),
        (net.res_trafo, 'trafo', 'hv', ('p_hv_mw', 'q_hv_mvar')),
    ):
        for k in table.index:
            for kind, col in zip(('p', 'q'), cols, strict=True):
                value = table[col][k]
                s = max(SIGMA_P * abs(value), FLOOR)
                _add(net, kind, element_type, value + noise(s), s, k, side)


@contextmanager
def quiet_estimation():
    """Silence two warnings raised inside pandapower's estimation code, not by this module.

    Its DataFrame handling trips pandas' SettingWithCopyWarning, and ``results.py`` negates a bool with ``~``, which
    Python 3.14 reports as deprecated. Neither changes the estimate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.SettingWithCopyWarning)
        warnings.filterwarnings('ignore', "Bitwise inversion '~' on bool is deprecated", DeprecationWarning)
        yield


def estimate(net: pp.pandapowerNet) -> bool:
    """Run the WLS estimator from a flat start; True when it converged."""
    with quiet_estimation():
        return bool(est.estimate(net, init='flat')['success'])


def chi2(net: pp.pandapowerNet) -> bool:
    """True when the chi-square test suspects bad data."""
    with quiet_estimation():
        return bool(est.chi2_analysis(net, init='flat'))


def remove_bad_data(net: pp.pandapowerNet) -> set[int]:
    """Largest-normalized-residual removal; returns the indices of the measurements it dropped."""
    before = set(net.measurement.index)
    with quiet_estimation():
        est.remove_bad_data(net, init='flat', rn_max_threshold=3.0)
    return before - set(net.measurement.index)


def estimation_errors(net: pp.pandapowerNet) -> tuple[float, float]:
    """Largest voltage magnitude (pu) and angle (degree) error of the estimate against the power flow."""
    dv = float(np.abs(net.res_bus_est.vm_pu - net.res_bus.vm_pu).max())
    da = float(np.abs(net.res_bus_est.va_degree - net.res_bus.va_degree).max())
    return dv, da


def normalized_residuals(net: pp.pandapowerNet) -> pd.DataFrame:
    """Measurement table with the estimated value and the residual divided by the standard deviation, largest first."""
    scale = net.res_bus_est.vm_pu.reindex(net.shunt.bus).to_numpy() ** 2 * net.shunt.step.to_numpy()
    shunt_p = pd.Series(net.shunt.p_mw.to_numpy() * scale, index=net.shunt.bus).groupby(level=0).sum()
    shunt_q = pd.Series(net.shunt.q_mvar.to_numpy() * scale, index=net.shunt.bus).groupby(level=0).sum()
    est_tables = {
        ('bus', 'v'): net.res_bus_est.vm_pu,
        ('bus', 'p'): net.res_bus_est.p_mw - shunt_p.reindex(net.bus.index, fill_value=0.0),
        ('bus', 'q'): net.res_bus_est.q_mvar - shunt_q.reindex(net.bus.index, fill_value=0.0),
        ('line', 'p'): net.res_line_est.p_from_mw,
        ('line', 'q'): net.res_line_est.q_from_mvar,
        ('trafo', 'p'): net.res_trafo_est.p_hv_mw,
        ('trafo', 'q'): net.res_trafo_est.q_hv_mvar,
    }
    m = net.measurement[['measurement_type', 'element_type', 'element', 'value', 'std_dev']].copy()
    m['estimate'] = [est_tables[(r.element_type, r.measurement_type)][r.element] for r in m.itertuples()]
    m['r_n'] = (m.value - m.estimate) / m.std_dev
    return m.reindex(m.r_n.abs().sort_values(ascending=False).index)


def corrupt_first_line_flow(net: pp.pandapowerNet, factor: float = 1.3) -> int:
    """Scale the first active-power line measurement by ``factor``; return its index."""
    idx = net.measurement.index[(net.measurement.measurement_type == 'p') & (net.measurement.element_type == 'line')][
        0
    ]
    net.measurement.loc[idx, 'value'] *= factor
    return int(idx)


def main() -> None:
    """Exact and noisy estimates, the bad datum, and the topology error with its residual pattern."""
    net = solved_case14()
    add_measurements(net, None)
    estimate(net)
    dv, da = estimation_errors(net)
    print(f'exact measurements: max |dV| {dv:.1e} pu, max |d angle| {da:.1e} deg, chi2 suspects bad data: {chi2(net)}')

    net = solved_case14()
    add_measurements(net, np.random.default_rng(14))
    estimate(net)
    dv, da = estimation_errors(net)
    print(f'{len(net.measurement)} noisy measurements, max |dV| {dv:.4f} pu, |d angle| {da:.3f} deg; chi2 {chi2(net)}')

    idx = corrupt_first_line_flow(net)
    print(f'measurement {idx} (p of line 0, from side) scaled by 1.3: chi2 suspects bad data: {chi2(net)}')
    estimate(net)
    print('largest normalized residuals:')
    print(normalized_residuals(net).head(3).round(2).to_string())
    removed = remove_bad_data(net)
    print(f'remove_bad_data dropped measurements {sorted(removed)}; the corrupted one is {idx}; chi2 now: {chi2(net)}')

    net = solved_case14()
    add_measurements(net, np.random.default_rng(14))
    net.line.loc[8, 'in_service'] = False  # the model opens line 8 (buses 5-11), the measurements still see it closed
    estimate(net)
    print(
        f'topology error on line 8 ({net.line.from_bus[8]}-{net.line.to_bus[8]}): chi2 suspects bad data: {chi2(net)}'
    )
    print(normalized_residuals(net).head(6).round(2).to_string())


if __name__ == '__main__':
    main()
