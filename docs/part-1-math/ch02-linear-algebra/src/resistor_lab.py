"""The resistor network of the Chapter 2 lab: Laplacian, null space, nodal solve, rank-1 update (Chapter 2).

Lab steps 1 to 5 on the six-node network of ``lab_network.py``, and the numeric checks of Exercises 1, 2, 4 and 5.
The incidence matrix, the Laplacian and the grounded solve come from ``laplacian.py``; this module adds what the lab
asks on top of them. ``python resistor_lab.py`` prints every number the solutions quote; the eigenvalue plot of
step 2 is built by ``eigenvalue_figure`` and described, not shown.
"""

import time

import numpy as np
from lab_network import fr, g, to
from laplacian import incidence, laplacian, nodal_voltages
from scipy.linalg import null_space

N_NODES = 6
FR, TO, G = np.array(fr), np.array(to), np.array(g, dtype=float)
INJECTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # 1 A into node 0, out at node 5 (step 3)
ZERO_TOL = 1e-9


def build(fr: np.ndarray, to: np.ndarray, g: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Incidence matrix A and weighted Laplacian G = Aᵀ diag(g) A of a branch list (step 1)."""
    a = incidence(fr, to, n)
    return a, laplacian(a, g)


def checks(gmat: np.ndarray) -> dict[str, float | int | bool]:
    """Symmetry, row sums and the number of zero eigenvalues of a Laplacian (step 1)."""
    w = np.linalg.eigvalsh(gmat)
    return {
        'symmetric': bool(np.allclose(gmat, gmat.T)),
        'max_row_sum': float(np.abs(gmat.sum(axis=1)).max()),
        'zero_eigenvalues': int(np.sum(np.abs(w) < ZERO_TOL)),
        'eigenvalues': np.round(w, 4),
    }


def without_branch(fr: np.ndarray, to: np.ndarray, g: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The branch list without branch ``k`` (step 2 deletes branch (4, 5), the last one)."""
    keep = np.arange(len(fr)) != k
    return fr[keep], to[keep], g[keep]


def components(gmat: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Orthonormal null-space basis of G and the connected components it reveals (step 2).

    Two nodes are in the same component when every basis vector takes the same value on both: the null space of a
    Laplacian is spanned by the indicator vectors of the components.
    """
    basis = null_space(gmat)
    groups: dict[tuple, list[int]] = {}
    for node, row in enumerate(np.round(basis, 8)):
        groups.setdefault(tuple(row), []).append(node)
    return basis, sorted(groups.values())


def branch_currents(a: np.ndarray, g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Currents g ⊙ (A v), positive from the from-node to the to-node."""
    return g * (a @ v)


def solve(a: np.ndarray, g: np.ndarray, i: np.ndarray, ground: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Node voltages with ``ground`` at 0 V, branch currents and the KCL residual max|Aᵀ(g ⊙ A v) − i| (steps 3, 4)."""
    gmat = laplacian(a, g)
    v = nodal_voltages(gmat, i, ground)
    cur = branch_currents(a, g, v)
    return v, cur, float(np.abs(a.T @ cur - i).max())


def rank_one_update(gmat: np.ndarray, a_row: np.ndarray, g_new: float) -> np.ndarray:
    """G + g a aᵀ, the Laplacian after adding one branch whose incidence row is ``a_row`` (step 5)."""
    return gmat + g_new * np.outer(a_row, a_row)


def eigenvalue_figure(full: np.ndarray, cut: np.ndarray):
    """Matplotlib figure with the sorted eigenvalues of the full and the cut network (step 2)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(np.sort(full), 'o-', label='full network (7 branches)')
    ax.plot(np.sort(cut), 's--', label='without branch (4, 5)')
    ax.set_xlabel('index')
    ax.set_ylabel('eigenvalue of G (S)')
    ax.legend()
    return fig


def triangle() -> dict[str, np.ndarray | int]:
    """Incidence matrix, Aᵀ A and null space of a triangle, with and without one branch (Exercises 1 and 2)."""
    a3 = incidence(np.array([0, 1, 2]), np.array([1, 2, 0]), 3)
    a2 = a3[:2]
    return {
        'A': a3,
        'AtA': a3.T @ a3,
        'null': null_space(a3.T @ a3),
        'AtA_without_branch': a2.T @ a2,
        'null_without_branch': null_space(a2.T @ a2),
        'rank': int(np.linalg.matrix_rank(a3.T @ a3)),
        'rank_without_branch': int(np.linalg.matrix_rank(a2.T @ a2)),
    }


def quadratic_form(
    gmat: np.ndarray, fr: np.ndarray, to: np.ndarray, g: np.ndarray, x: np.ndarray
) -> tuple[float, float]:
    """xᵀ G x and Σ g_k (x_fk − x_tk)², which Exercise 2 proves equal."""
    return float(x @ gmat @ x), float(np.sum(g * (x[fr] - x[to]) ** 2))


def three_ways(a: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aᵀ diag(g) A with an explicit diagonal matrix, with broadcasting and with einsum (Exercise 4)."""
    return a.T @ np.diag(g) @ a, a.T @ (g[:, None] * a), np.einsum('ki,k,kj->ij', a, g, a)


def random_network(n: int, extra: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A connected branch list on ``n`` nodes: a chain plus ``extra`` random branches, conductances in (0.5, 1.5)."""
    rng = np.random.default_rng(seed)
    fr_ = np.concatenate([np.arange(n - 1), rng.integers(0, n, extra)])
    to_ = np.concatenate([np.arange(1, n), rng.integers(0, n, extra)])
    loops = fr_ == to_
    to_[loops] = (to_[loops] + 1) % n
    return fr_, to_, rng.uniform(0.5, 1.5, len(fr_))


def time_three_ways(n: int = 2000, extra: int = 1000) -> dict[str, float]:
    """Seconds taken by each of the three products on a random network with ``n`` nodes (Exercise 4).

    A fourth entry times ``einsum`` with ``optimize=True``, which contracts pairwise like the broadcast version
    instead of running one triple loop over k, i and j.
    """
    fr_, to_, g_ = random_network(n, extra)
    a = incidence(fr_, to_, n)
    out = {}
    for name, fn in (
        ('np.diag', lambda: a.T @ np.diag(g_) @ a),
        ('broadcast', lambda: a.T @ (g_[:, None] * a)),
        ('einsum', lambda: np.einsum('ki,k,kj->ij', a, g_, a)),
        ('einsum optimize=True', lambda: np.einsum('ki,k,kj->ij', a, g_, a, optimize=True)),
    ):
        start = time.perf_counter()
        fn()
        out[name] = time.perf_counter() - start
    return out


def rank_one_product(
    gmat: np.ndarray, a_row: np.ndarray, g_new: float, x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """G' x computed as (G + g a aᵀ) x and as G x + g (aᵀx) a, which Exercise 5 shows are equal."""
    return rank_one_update(gmat, a_row, g_new) @ x, gmat @ x + g_new * (a_row @ x) * a_row


def main() -> None:
    """Print the numbers of Lab steps 1 to 5 and of Exercises 1, 2, 4 and 5."""
    np.set_printoptions(precision=4, suppress=True, linewidth=119)
    a, gmat = build(FR, TO, G, N_NODES)
    print('Step 1:', checks(gmat))
    print('G =\n', gmat)
    fr_c, to_c, g_c = without_branch(FR, TO, G, 6)
    a_c, g_cut = build(fr_c, to_c, g_c, N_NODES)
    print('Step 2:', checks(g_cut))
    basis, comps = components(g_cut)
    print('null space basis (rows = nodes):\n', basis, '\ncomponents', comps)
    fig = eigenvalue_figure(np.linalg.eigvalsh(gmat), np.linalg.eigvalsh(g_cut))
    print('figure with', len(fig.axes[0].lines), 'curves')
    v0, cur0, res0 = solve(a, G, INJECTION, ground=0)
    print('Step 3: v (ground 0) =', v0, '\n  currents =', cur0, ' KCL residual', f'{res0:.1e}')
    v5, cur5, res5 = solve(a, G, INJECTION, ground=5)
    print(
        'Step 4: v (ground 5) =',
        v5,
        '\n  shift v5 - v0 =',
        v5 - v0,
        '\n  currents =',
        cur5,
        ' KCL residual',
        f'{res5:.1e}',
    )
    a_new = np.array([1.0, 0, 0, 0, 0, -1.0])
    g_plus = rank_one_update(gmat, a_new, 1.0)
    a_plus, g_plus_direct = build(np.append(FR, 0), np.append(TO, 5), np.append(G, 1.0), N_NODES)
    print('Step 5: rank-1 update equals rebuilt G:', np.allclose(g_plus, g_plus_direct))
    v_plus, cur_plus, _ = solve(a_plus, np.append(G, 1.0), INJECTION, ground=0)
    print('  v =', v_plus, '\n  currents (old branches) =', cur_plus[:7], ' new branch', f'{cur_plus[7]:.4f}')
    print('  change in old currents =', cur_plus[:7] - cur0)
    tri = triangle()
    print('Exercise 1: AtA =\n', tri['AtA'], '\n  rank', tri['rank'], ' null =', tri['null'].ravel())
    print(
        '  without one branch: AtA =\n',
        tri['AtA_without_branch'],
        '\n  rank',
        tri['rank_without_branch'],
        ' null =',
        tri['null_without_branch'].ravel(),
    )
    x = np.random.default_rng(1).normal(size=N_NODES)
    print('Exercise 2: x^T G x, sum g (x_f - x_t)^2 =', quadratic_form(gmat, FR, TO, G, x))
    g1, g2, g3 = three_ways(a, G)
    print('Exercise 4: three ways agree:', np.allclose(g1, g2) and np.allclose(g2, g3))
    print('  times for 2000 nodes:', {k: f'{t:.3f} s' for k, t in time_three_ways().items()})
    lhs, rhs = rank_one_product(gmat, a_new, 1.0, x)
    print("Exercise 5: max|G'x - (Gx + g (a^T x) a)| =", f'{np.abs(lhs - rhs).max():.1e}')


if __name__ == '__main__':
    main()
