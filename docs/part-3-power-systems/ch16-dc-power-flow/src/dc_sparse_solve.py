"""Sparse DC power flow of IEEE 14 for many injection snapshots at once (Chapter 16, Exercise 3).

Bbus and Bf are assembled as ``scipy.sparse`` matrices from the branch table that pandapower builds for ``case14``,
factorized once with ``splu`` and reused for a ``(n_timesteps, n_bus)`` block of balanced injections.
"""

import timeit

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from dc_vs_ac import dc_case


def susceptance_matrices(
    frm: np.ndarray, to: np.ndarray, b: np.ndarray, n_bus: int
) -> tuple[sps.csc_matrix, sps.csr_matrix]:
    """Sparse Bbus = Aᵀ diag(b) A and Bf = diag(b) A for branch susceptances ``b``."""
    m = len(frm)
    rows = np.concatenate([np.arange(m), np.arange(m)])
    a = sps.csr_matrix(
        (np.concatenate([np.ones(m), -np.ones(m)]), (rows, np.concatenate([frm, to]))), shape=(m, n_bus)
    )
    bf = sps.diags(b) @ a
    return (a.T @ bf).tocsc(), bf.tocsr()


def flows_many(lu: spla.SuperLU, bf_reduced: sps.csr_matrix, p: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Flows ``(n_timesteps, n_branches)`` for the injections ``p`` ``(n_timesteps, n_bus)`` with one factorization."""
    theta = lu.solve(p[:, keep].T)  # (n_bus - 1, n_timesteps), every snapshot in one call
    return (bf_reduced @ theta).T


def flows_one_by_one(
    bbus_reduced: sps.csc_matrix, bf_reduced: sps.csr_matrix, p: np.ndarray, keep: np.ndarray
) -> np.ndarray:
    """The same flows with a separate ``spsolve`` (and therefore a separate factorization) per snapshot."""
    return np.array([bf_reduced @ spla.spsolve(bbus_reduced, row[keep]) for row in p])


def main(n_timesteps: int = 1000) -> None:
    """Build IEEE 14, draw balanced snapshots, time the two approaches and check that they agree."""
    _, table, p0, slack = dc_case('case14')
    n_bus = int(max(table['from'].max(), table['to'].max())) + 1
    b = 1 / (table['x'].to_numpy() * table['tap'].to_numpy())
    bbus, bf = susceptance_matrices(table['from'].to_numpy(), table['to'].to_numpy(), b, n_bus)
    keep = np.arange(n_bus) != slack
    bbus_r, bf_r = bbus[keep][:, keep], bf[:, keep]
    rng = np.random.default_rng(16)
    p = p0 + rng.normal(scale=10.0, size=(n_timesteps, n_bus))
    p[0] = p0  # snapshot 0 is the base case itself
    p[:, slack] -= p.sum(axis=1)  # the slack balances every snapshot
    print(f'IEEE 14: Bbus {bbus.shape} with {bbus.nnz} nonzeros, Bf {bf.shape}; injections {p.shape}')
    lu = spla.splu(bbus_r)
    many = flows_many(lu, bf_r, p, keep)
    single = flows_one_by_one(bbus_r, bf_r, p, keep)
    base_err = np.abs(many[0] - table['p_pandapower'].to_numpy()).max()
    print(f'max |block solve - separate solves| = {np.abs(many - single).max():.1e} MW; ', end='')
    print(f'snapshot 0 (the base case) matches rundcpp to {base_err:.1e} MW')
    t_many = min(timeit.repeat(lambda: flows_many(spla.splu(bbus_r), bf_r, p, keep), number=1, repeat=5))
    t_single = min(timeit.repeat(lambda: flows_one_by_one(bbus_r, bf_r, p, keep), number=1, repeat=3))
    print(f'{n_timesteps} snapshots: one factorization + block solve {1e3 * t_many:.2f} ms; ', end='')
    print(f'{n_timesteps} separate solves {1e3 * t_single:.1f} ms; ratio {t_single / t_many:.0f}x')


if __name__ == '__main__':
    main()
