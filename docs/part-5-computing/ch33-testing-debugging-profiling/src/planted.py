"""The Chapter 33 lab in miniature: a DC kernel on IEEE 14 with three planted bugs and the checks that catch them.

The bugs mirror the instructor's list: ``psdf_sign`` flips the phase-shift columns, ``float32`` stores the PTDF in
single precision (what skipping ``jax_enable_x64`` does to ToOp's arrays) and ``wrong_slack`` reduces the
susceptance matrix at the wrong bus. ``python planted.py`` prints the symptom table, the localization table and
the overload energies the solution quotes.
"""

import dcgrid
import numpy as np
import pandapower.networks as pn

BUGS = ('psdf_sign', 'float32', 'wrong_slack')
CHECKS = (
    'oracle, signed flows, atol 1e-6 MW',
    'oracle, |flows|, atol 1e-6 MW',
    'oracle, signed flows, atol 1e-2 MW',
    'oracle at zero shift, atol 1e-6 MW',
    'slack column of the PTDF is zero',
    'LODF N-1 equals re-solve, atol 1e-8',
)


def ieee14(shift_degree=10.0):
    """IEEE 14 from pandapower with the first transformer phase-shifted, and pandapower's own DC flows."""
    net = pn.case14()
    net.trafo['shift_degree'] = 0.0
    net.trafo.loc[net.trafo.index[0], 'shift_degree'] = shift_degree
    return dcgrid.pandapower_dc(net)


def ptdf(data, bug=None):
    """The PTDF of ``data``, with the ``float32`` and ``wrong_slack`` bugs where asked."""
    slack = data.slack + 1 if bug == 'wrong_slack' else data.slack
    h = dcgrid.ptdf_matrix(data.fr, data.to, data.b, data.n_bus, slack)
    return h.astype(np.float32).astype(np.float64) if bug == 'float32' else h


def psdf(data, h, bug=None):
    """PSDF columns in MW per radian for the shifted branches, and their indices; ``psdf_sign`` flips the sign.

    A shift of phi on branch k acts like an injection pair -b_k phi at its ends plus -b_k phi on the branch itself,
    so the column is b_k (PTDF[:, f_k] - PTDF[:, t_k]) with b_k subtracted from entry k (MATPOWER's convention).
    """
    k = np.flatnonzero(data.shift != 0.0)
    cols = data.b[k] * (h[:, data.fr[k]] - h[:, data.to[k]])
    cols[k, np.arange(len(k))] -= data.b[k]
    cols *= data.base_mva
    return (-cols if bug == 'psdf_sign' else cols), k


def n0_flows(data, bug=None):
    """Base-case flows in MW: PTDF * p + PSDF * shift."""
    h = ptdf(data, bug)
    s, k = psdf(data, h, bug)
    return h @ data.p + s @ data.shift[k]


def n1_flows(data, bug=None):
    """LODF-based post-contingency flows (one row per outaged branch) and the outages that are not bridges."""
    f0 = n0_flows(data, bug)
    lodf, den = dcgrid.lodf_matrix(ptdf(data, bug), data.fr, data.to)
    outages = np.flatnonzero(np.abs(den) > 1e-11)
    return f0[None, :] + lodf[:, outages].T * f0[outages, None], outages


def resolved_n1_flows(data, bug=None):
    """Post-contingency flows by solving the grid again without each non-bridge branch (the reference)."""
    _, outages = n1_flows(data, bug)
    rows = np.zeros((len(outages), len(data.fr)))
    for i, o in enumerate(outages):
        keep = np.arange(len(data.fr)) != o
        rows[i, keep] = n0_flows(data.without(o), bug)
    return rows, outages


def limits(data):
    """Branch limits in MW that leave some N-1 overloads: 5 % above the correct base-case flows, at least 20 MW."""
    return np.maximum(1.05 * np.abs(n0_flows(data)), 20.0)


def overload_energy_n1(data, bug=None):
    """ToOp's N-1 overload energy: per branch the worst contingency overload, summed over branches."""
    f1, _ = n1_flows(data, bug)
    return float(np.maximum(np.abs(f1) - limits(data), 0.0).max(axis=0).sum())


def symptoms(data, bug=None):
    """Which checks pass (True) or fail (False) for one planted bug: the lab's step 1."""
    f0 = n0_flows(data, bug)
    zero = ieee14(0.0)
    f1, _ = n1_flows(data, bug)
    f1_ref, _ = resolved_n1_flows(data, bug)
    results = (
        np.allclose(f0, data.flows, rtol=0.0, atol=1e-6),
        np.allclose(np.abs(f0), np.abs(data.flows), rtol=0.0, atol=1e-6),
        np.allclose(f0, data.flows, rtol=0.0, atol=1e-2),
        np.allclose(n0_flows(zero, bug), zero.flows, rtol=0.0, atol=1e-6),
        bool(np.all(ptdf(data, bug)[:, data.slack] == 0.0)),
        np.allclose(f1, f1_ref, rtol=0.0, atol=1e-8),
    )
    return dict(zip(CHECKS, results, strict=True))


def localize(data, bug=None):
    """Largest difference of each intermediate array against the correct code: the lab's step 2."""
    h, h_ok = ptdf(data, bug), ptdf(data)
    s, _ = psdf(data, h_ok, bug)
    s_ok, _ = psdf(data, h_ok)
    return {
        'PTDF': float(np.abs(h - h_ok).max()),
        'PSDF (from the correct PTDF)': float(np.abs(s - s_ok).max()),
        'N-0 flows at zero shift': float(np.abs(n0_flows(ieee14(0.0), bug) - ieee14(0.0).flows).max()),
        'N-0 flows at 10 degrees': float(np.abs(n0_flows(data, bug) - data.flows).max()),
    }


def main():
    """Print the tables of the lab solution."""
    data = ieee14()
    print(
        f'IEEE 14: {len(data.fr)} branches, {data.n_bus} buses, slack {data.slack}, shifted branch(es) '
        f'{np.flatnonzero(data.shift != 0.0).tolist()} at {np.rad2deg(data.shift[data.shift != 0.0])} degrees'
    )
    print(
        f'largest |flow| {np.abs(data.flows).max():.2f} MW, correct overload energy N-1 '
        f'{overload_energy_n1(data):.3f} MW'
    )
    print('\nStep 1, symptoms (True = the check passes):')
    print(f'{"check":42s}' + ''.join(f'{b:>13s}' for b in ('correct', *BUGS)))
    tables = {bug: symptoms(data, bug) for bug in (None, *BUGS)}
    for check in CHECKS:
        print(f'{check:42s}' + ''.join(f'{tables[bug][check]!s:>13s}' for bug in (None, *BUGS)))
    print('\noverload energy N-1 per bug (MW):', {str(bug): round(overload_energy_n1(data, bug), 3) for bug in BUGS})
    print('\nStep 2, localization (largest difference against the correct code):')
    for bug in BUGS:
        print(f'  {bug:12s}', {k: f'{v:.2e}' for k, v in localize(data, bug).items()})


if __name__ == '__main__':
    main()
