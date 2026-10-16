"""Rank-1 updates of a resistor network: denominators, three ways to remove a branch, timings (Chapter 4, Lab 1–3).

The network, the reduced incidence matrix and the Sherman–Morrison update come from ``sherman_morrison.py``;
this module adds the lab's tables, the direct update of the solution and the comparison against refactorization
on a larger grid.
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sherman_morrison import fr, g, n, reduced_incidence, remove_branch, to

INJECTION_NODE = 5  # 1 A injected here and withdrawn at the grounded node 0


def injection(n_nodes: int = n, node: int = INJECTION_NODE, ground: int = 0) -> np.ndarray:
    """Reduced injection vector: 1 A at ``node``, the grounded node's row removed."""
    i = np.zeros(n_nodes)
    i[node] = 1.0
    return i[np.arange(n_nodes) != ground]


def denominator_table(
    ar: np.ndarray, cond: np.ndarray, minv: np.ndarray, fr_: np.ndarray, to_: np.ndarray
) -> pd.DataFrame:
    """Per branch: ends, conductance, gₖ R_eff,k and the Sherman–Morrison denominator 1 − gₖ R_eff,k (lab step 1)."""
    rows = []
    for k in range(len(cond)):
        g_reff = cond[k] * (ar[k] @ minv @ ar[k])
        rows.append(
            {'branch': k, 'from': fr_[k], 'to': to_[k], 'g': cond[k], 'g_reff': g_reff, 'denominator': 1 - g_reff}
        )
    return pd.DataFrame(rows)


def update_solution(v: np.ndarray, minv: np.ndarray, ar: np.ndarray, cond: np.ndarray, k: int) -> np.ndarray:
    """Node voltages after removing branch ``k`` without forming a new inverse (lab step 2c).

    v' = v + Minv aₖ · gₖ (aₖᵀ v) / (1 − gₖ aₖᵀ Minv aₖ): the flow that branch k carried, gₖ aₖᵀ v, is re-injected at
    its ends and spread with the old inverse, scaled by the same denominator as the inverse update.
    """
    a = ar[k]
    u = minv @ a
    den = 1 - cond[k] * (a @ u)
    return v + u * (cond[k] * (a @ v) / den)


def remove_three_ways(k: int) -> dict[str, np.ndarray]:
    """Voltages after removing branch ``k`` by (a) rebuilding, (b) updating the inverse, (c) updating the solution."""
    ar = reduced_incidence(fr, to, n)
    gr = ar.T @ np.diag(g) @ ar
    minv = np.linalg.inv(gr)
    i_r = injection()
    v = minv @ i_r
    keep = np.arange(len(g)) != k
    rebuilt = np.linalg.solve(ar[keep].T @ np.diag(g[keep]) @ ar[keep], i_r)
    inverse = remove_branch(minv, ar, g, k) @ i_r
    direct = update_solution(v, minv, ar, g, k)
    return {'before': v, 'rebuild': rebuilt, 'inverse_update': inverse, 'solution_update': direct}


def grid_network(side: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Edges of a ``side × side`` grid graph with random conductances in [0.5, 2) S (lab step 3)."""
    graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    edges = np.array(graph.edges())
    rng = np.random.default_rng(seed)
    return edges[:, 0], edges[:, 1], rng.uniform(0.5, 2.0, len(edges)), graph.number_of_nodes()


def sparse_reduced_laplacian(fr_: np.ndarray, to_: np.ndarray, cond: np.ndarray, n_nodes: int) -> sp.csc_matrix:
    """Reduced Laplacian Aᵀ diag(g) A in CSC form, node 0 grounded."""
    m = len(cond)
    rows = np.r_[np.arange(m), np.arange(m)]
    cols = np.r_[fr_, to_]
    vals = np.r_[np.ones(m), -np.ones(m)]
    a = sp.csc_matrix((vals, (rows, cols)), shape=(m, n_nodes))[:, 1:]
    return (a.T @ sp.diags(cond) @ a).tocsc()


def time_removals(
    fr_: np.ndarray, to_: np.ndarray, cond: np.ndarray, n_nodes: int, n_removals: int = 100, seed: int = 1
) -> pd.DataFrame:
    """Time 'rebuild and solve' against 'factor once, then update the solution' for single-branch removals.

    The update needs Minv aₖ, which the stored factorization supplies with one triangular solve pair, so no inverse
    is formed. Both methods return the voltages after each removal; the table carries the per-removal seconds and
    the largest disagreement between them.
    """
    rng = np.random.default_rng(seed)
    gr = sparse_reduced_laplacian(fr_, to_, cond, n_nodes)
    i_r = injection(n_nodes, node=n_nodes - 1)
    m = len(cond)
    a_rows = sp.csc_matrix(
        (np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[fr_, to_])), shape=(m, n_nodes)
    )[:, 1:].tocsr()
    t0 = time.perf_counter()
    lu = spla.splu(gr)
    v = lu.solve(i_r)
    t_factor = time.perf_counter() - t0
    rows = []
    for k in rng.choice(m, size=n_removals, replace=False):
        keep = np.arange(m) != k
        t0 = time.perf_counter()
        v_rebuild = spla.spsolve(sparse_reduced_laplacian(fr_[keep], to_[keep], cond[keep], n_nodes), i_r)
        t_rebuild = time.perf_counter() - t0
        t0 = time.perf_counter()
        a = a_rows[k].toarray().ravel()
        u = lu.solve(a)
        den = 1 - cond[k] * (a @ u)
        v_update = v + u * (cond[k] * (a @ v) / den) if abs(den) > 1e-12 else np.full_like(v, np.nan)
        t_update = time.perf_counter() - t0
        rows.append(
            {
                'branch': int(k),
                'rebuild_s': t_rebuild,
                'update_s': t_update,
                'max_diff': float(np.max(np.abs(v_rebuild - v_update))) if np.isfinite(den) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs['factor_s'] = t_factor
    return out


def plot_timings(table: pd.DataFrame) -> matplotlib.figure.Figure:
    """Cumulative seconds of both methods against the number of removals (lab step 3)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(1, len(table) + 1)
    ax.plot(x, table['rebuild_s'].cumsum(), label='rebuild and solve')
    ax.plot(x, table.attrs.get('factor_s', 0) + table['update_s'].cumsum(), label='factor once + Sherman–Morrison')
    ax.set_xlabel('single-branch removals')
    ax.set_ylabel('cumulative seconds')
    ax.legend()
    return fig


def main() -> None:
    """Print the denominator table, the three-way check and the timing summary quoted in the solutions."""
    ar = reduced_incidence(fr, to, n)
    minv = np.linalg.inv(ar.T @ np.diag(g) @ ar)
    pd.set_option('display.precision', 6)
    print(denominator_table(ar, g, minv, fr, to).to_string(index=False))
    ways = remove_three_ways(3)
    print('voltages before removing (1, 3):', np.round(ways['before'], 6))
    print('after, rebuilt                 :', np.round(ways['rebuild'], 6))
    for name in ('inverse_update', 'solution_update'):
        print(f'max |{name} - rebuild| = {np.max(np.abs(ways[name] - ways["rebuild"])):.1e}')
    table = time_removals(*grid_network())
    total_rebuild, total_update = table['rebuild_s'].sum(), table.attrs['factor_s'] + table['update_s'].sum()
    print(
        f'40x40 grid, {len(table)} removals: rebuild {total_rebuild:.3f} s, factor once + update {total_update:.3f} s'
    )
    print(f'speed-up {total_rebuild / total_update:.1f}x, max disagreement {table["max_diff"].max():.1e}')


if __name__ == '__main__':
    main()
