"""LODF of IEEE 118 in float32 and float64: which outages lose accuracy, and why (Chapter 31, Exercise 4)."""

import jax.numpy as jnp
import numpy as np
from grid_data import branch_data
from n1_screen import ptdf_matrix


def lodf_matrix(ptdf, from_bus, to_bus):
    """LODF (monitored branch x outaged branch) in the dtype of ``ptdf``, plus the LODF denominators.

    LODF[a, b] = (PTDF[a, f_b] - PTDF[a, t_b]) / (1 - (PTDF[b, f_b] - PTDF[b, t_b])), LODF[b, b] = -1.
    Bridges have a zero denominator and an infinite column; the caller masks them with ``abs(denom) > 1e-11``.
    """
    col = ptdf[:, from_bus] - ptdf[:, to_bus]  # (n_branch, n_outage)
    denom = 1.0 - jnp.diagonal(col)
    lodf = col / denom[None, :]
    eye = jnp.eye(ptdf.shape[0], dtype=bool)
    return jnp.where(eye, -1.0, lodf), denom


def precision_study(case: str = 'case118') -> dict:
    """Per outage: |denominator| and the largest |LODF32 - LODF64| in its column (bridges excluded)."""
    from_bus, to_bus, x, p = (jnp.array(v) for v in branch_data(case))
    ptdf64 = ptdf_matrix(from_bus, to_bus, x, len(p))
    lodf64, denom = lodf_matrix(ptdf64, from_bus, to_bus)
    lodf32, _ = lodf_matrix(ptdf64.astype(jnp.float32), from_bus, to_bus)
    ok = np.abs(np.asarray(denom)) > 1e-11
    error = np.full(len(ok), np.nan)  # bridge columns are infinite in both precisions: no error to compare
    error[ok] = np.abs(np.asarray(lodf32, dtype=np.float64)[:, ok] - np.asarray(lodf64)[:, ok]).max(axis=0)
    return {
        'from_bus': np.asarray(from_bus),
        'to_bus': np.asarray(to_bus),
        'abs_denom': np.abs(np.asarray(denom)),
        'error': error,
        'ok': ok,
    }


def main() -> None:
    """Print the bridges, the five outages with the smallest denominators and the error trend."""
    study = precision_study()
    ok, denom, error = study['ok'], study['abs_denom'], study['error']
    print(f'{int((~ok).sum())} bridges (|1 - (PTDF_b,fb - PTDF_b,tb)| < 1e-11), excluded from the comparison')
    order = np.argsort(denom + np.where(ok, 0.0, np.inf))[:5]
    print('smallest denominators among non-bridges:')
    for b in order:
        print(
            f'  outage {b:3d} (bus {study["from_bus"][b] + 1:3d}-{study["to_bus"][b] + 1:3d}): '
            f'|denom| = {denom[b]:.4f}, max |LODF32 - LODF64| = {error[b]:.2e}'
        )
    print(f'float32 error over all non-bridge outages: median {np.median(error[ok]):.1e}, max {error[ok].max():.1e}')
    small, large = np.median(error[ok][denom[ok] < 0.2]), np.median(error[ok][denom[ok] >= 0.5])
    print(f'median error, |denom| < 0.2: {small:.1e}; |denom| >= 0.5: {large:.1e}')
    rho = np.corrcoef(np.log10(denom[ok]), np.log10(error[ok]))[0, 1]
    print(f'correlation of log10 error with log10 |denom|: {rho:.2f}')


if __name__ == '__main__':
    main()
