"""N-1 flows under ``jax.jit`` on a grid with a bridge: silent NaN, ``jax_debug_nans`` and ``jax.debug.print`` (Ex. 2).

The grid is a five-node ring with a sixth node hung on node 0 through a single branch, the bridge. Its LODF
denominator is exactly zero, and the three ways of running the same jitted function show three behaviors.
"""

from functools import partial

import dcgrid
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update('jax_enable_x64', True)


def bridge_grid():
    """Branches (from, to), unit susceptances and injections; branch 5 (0-5) is the bridge."""
    fr = np.array([0, 1, 2, 3, 4, 0])
    to = np.array([1, 2, 3, 4, 0, 5])
    p = np.array([0.0, -0.5, 0.5, -0.5, 0.2, -0.5])  # node 0 is the slack and balances the rest
    return fr, to, np.ones(6), p


@partial(jax.jit, static_argnames='verbose')
def n1_flows(ptdf, fr, to, p, verbose=False):
    """Post-contingency flows, one row per outaged branch, through the LODF formula; no check on the denominator."""
    f0 = ptdf @ p
    col = ptdf[:, fr] - ptdf[:, to]
    den = 1.0 - jnp.diag(col)
    if verbose:
        jax.debug.print('LODF denominators: {d}', d=den)
    lodf = jnp.fill_diagonal(col / den, -1.0, inplace=False)
    return f0[None, :] + lodf.T * f0[:, None]


@jax.jit
def n1_flows_flagged(ptdf, fr, to, p):
    """The same flows plus a success flag per outage, the way ToOp's ``calc_lodf`` kernel reports a bridge."""
    f0 = ptdf @ p
    col = ptdf[:, fr] - ptdf[:, to]
    den = 1.0 - jnp.diag(col)
    lodf = jnp.fill_diagonal(col / den, -1.0, inplace=False)
    return f0[None, :] + lodf.T * f0[:, None], jnp.abs(den) > 1e-11


def run_plain():
    """Behavior 1: the function returns, with NaN where the bridge was outaged."""
    fr, to, b, p = bridge_grid()
    ptdf = dcgrid.ptdf_matrix(fr, to, b, 6)
    return np.asarray(n1_flows(ptdf, fr, to, p))


def run_debug_nans():
    """Behavior 2: with ``jax_debug_nans`` the same call raises FloatingPointError; returns the message."""
    fr, to, b, p = bridge_grid()
    ptdf = dcgrid.ptdf_matrix(fr, to, b, 6)
    jax.config.update('jax_debug_nans', True)
    # An executable compiled while the flag was off carries no NaN check and can be reused from the jit cache
    # (seen with JAX 0.11 once other jitted functions have run in the process): drop it so the call recompiles.
    jax.clear_caches()
    try:
        n1_flows(ptdf, fr, to, p).block_until_ready()
    except FloatingPointError as err:
        return str(err).splitlines()[0]
    finally:
        jax.config.update('jax_debug_nans', False)
    return 'no exception'


def run_with_print():
    """Behavior 3: ``jax.debug.print`` shows the denominators (0.0 for the bridge) and the result still has NaN."""
    fr, to, b, p = bridge_grid()
    ptdf = dcgrid.ptdf_matrix(fr, to, b, 6)
    return np.asarray(n1_flows(ptdf, fr, to, p, verbose=True))


def run_flagged():
    """Behavior 4: a success flag per outage, False for the bridge, and the caller masks that row."""
    fr, to, b, p = bridge_grid()
    ptdf = dcgrid.ptdf_matrix(fr, to, b, 6)
    flows, ok = n1_flows_flagged(ptdf, fr, to, p)
    return np.asarray(flows), np.asarray(ok)


def main():
    """Run the four variants and print what each one shows."""
    flows = run_plain()
    nan_rows = np.flatnonzero(np.isnan(flows).any(axis=1))
    print(f'plain jit: shape {flows.shape}, NaN in rows (outages) {nan_rows.tolist()}, no error raised')
    print('row 5 (bridge outaged):', flows[5])
    print('jax_debug_nans:', run_debug_nans())
    print('jax.debug.print of the denominators:')
    run_with_print()
    flows, ok = run_flagged()
    print(f'flagged: success per outage {ok.tolist()}; valid rows are finite: {np.isfinite(flows[ok]).all()}')


if __name__ == '__main__':
    main()
