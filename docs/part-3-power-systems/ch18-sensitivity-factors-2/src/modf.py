"""Multiple-outage distribution factors and the Woodbury susceptance update of Chapter 18, in NumPy.

Notation of the chapter: ``h`` is the PTDF (branches x nodes), ``frm``/``to`` the branch ends, and for a set of
branches O the matrix script-H[:, O] = h[:, frm_O] - h[:, to_O] holds the flow response to one MW sent across each
branch of O; its k x k block on the rows of O is script-H[O, O]. ``ieee14.load_case14()`` gives a grid to try them on.
"""

import numpy as np

ISLAND_TOL = 1e-10  # ToOp: |det| above this means the outage set leaves the grid connected


def transfer(h: np.ndarray, frm: np.ndarray, to: np.ndarray, branches: list[int]) -> np.ndarray:
    """script-H[:, O]: PTDF columns of the from-ends minus those of the to-ends (branches x k)."""
    return h[:, frm[branches]] - h[:, to[branches]]


def modf(h: np.ndarray, frm: np.ndarray, to: np.ndarray, outages: list[int]) -> tuple[np.ndarray, float]:
    """MODF (branches x k) of a joint outage of ``outages`` and det(I - script-H[O, O]).

    A determinant below ``ISLAND_TOL`` means the outage set islands the grid; the MODF is then all NaN.
    """
    n_t = transfer(h, frm, to, outages)
    m = np.eye(len(outages)) - n_t[outages]
    det = float(np.linalg.det(m))
    if abs(det) < ISLAND_TOL:
        return np.full_like(n_t, np.nan), det
    n_t = n_t.copy()
    n_t[outages] = -m  # the outaged branches lose their own flow
    return np.linalg.solve(m.T, n_t.T).T, det  # N (I - script-H[O, O])^{-1}


def apply_modf(h: np.ndarray, frm: np.ndarray, to: np.ndarray, outages: list[int]) -> np.ndarray:
    """PTDF after the joint outage: H + MODF H[O, :], rows O zero (equation 7 of Guo et al.)."""
    m, det = modf(h, frm, to, outages)
    if abs(det) < ISLAND_TOL:
        raise ValueError(f'the outage of {outages} islands the grid (det = {det:.2e})')
    h_new = h + m @ h[outages]
    h_new[outages] = 0.0
    return h_new


def modf_flows(f: np.ndarray, h: np.ndarray, frm: np.ndarray, to: np.ndarray, outages: list[int]) -> np.ndarray:
    """Flows after the joint outage: F + MODF F[O], with F[O] = 0 afterwards."""
    m, _ = modf(h, frm, to, outages)
    f_new = f + m @ f[outages]
    f_new[outages] = 0.0
    return f_new


def woodbury(h: np.ndarray, frm: np.ndarray, to: np.ndarray, branches: list[int], alpha: np.ndarray) -> np.ndarray:
    """PTDF after changing the susceptances of ``branches`` by the relative amounts ``alpha`` (delta b / b).

    H' = H + (E_O - script-H[:, O]) C with C = (I + diag(alpha) script-H[O, O])^{-1} diag(alpha) H[O, :]. With every
    alpha equal to -1 (susceptance to zero) it is the MODF update.
    """
    alpha = np.asarray(alpha, float)
    n_t = transfer(h, frm, to, branches)
    m = np.eye(len(branches)) + alpha[:, None] * n_t[branches]
    c = np.linalg.solve(m, alpha[:, None] * h[branches])
    e = np.zeros_like(n_t)
    e[branches, np.arange(len(branches))] = 1.0
    return h + (e - n_t) @ c


def lodf_matrix(h: np.ndarray, frm: np.ndarray, to: np.ndarray) -> np.ndarray:
    """LODF matrix (branches x outaged branches); columns of bridges are NaN; diagonal -1."""
    n_t = transfer(h, frm, to, list(range(h.shape[0])))
    den = 1.0 - np.diag(n_t)
    with np.errstate(divide='ignore', invalid='ignore'):
        lodf = np.where(np.abs(den) < 1e-11, np.nan, n_t / den)
    np.fill_diagonal(lodf, -1.0)
    return lodf


def modf_2x2(h: np.ndarray, frm: np.ndarray, to: np.ndarray, k1: int, k2: int) -> tuple[np.ndarray, float]:
    """Exercise 1: the MODF of two branches with the closed-form inverse of (I - script-H[O, O]).

    With a = script-H[k1, k1], b = script-H[k1, k2], c = script-H[k2, k1], d = script-H[k2, k2]:
    I - script-H[O, O] = [[1 - a, -b], [-c, 1 - d]] has determinant (1 - a)(1 - d) - b c and inverse
    [[1 - d, b], [c, 1 - a]] / det. Returns the MODF and the determinant.
    """
    n_t = transfer(h, frm, to, [k1, k2])
    a, b = n_t[k1]
    c, d = n_t[k2]
    det = (1 - a) * (1 - d) - b * c
    inv = np.array([[1 - d, b], [c, 1 - a]]) / det
    n_t = n_t.copy()
    n_t[[k1, k2]] = -np.array([[1 - a, -b], [-c, 1 - d]])
    return n_t @ inv, float(det)


def main() -> None:
    """Joint outage of two IEEE 14 branches and a 20 % susceptance change, each against a recomputed PTDF."""
    from ieee14 import load_case14

    grid = load_case14()
    h = grid.ptdf()
    out = [grid.names.index('L2-4'), grid.names.index('L4-5')]
    h_modf = apply_modf(h, grid.frm, grid.to, out)
    h_ref = grid.without(out).ptdf()
    keep = [k for k in range(grid.n_branch) if k not in out]
    print('MODF of L2-4 + L4-5: det =', round(modf(h, grid.frm, grid.to, out)[1], 6))
    print('  max |H_modf - H_recomputed| =', np.abs(h_modf[keep] - h_ref).max())
    print(
        '  max |H_woodbury(alpha=-1) - H_modf| =', np.abs(woodbury(h, grid.frm, grid.to, out, [-1, -1]) - h_modf).max()
    )
    t = grid.names.index('T5-6')
    h_w = woodbury(h, grid.frm, grid.to, [t], [0.2])
    print('T5-6 susceptance +20 %: max |H_woodbury - H_recomputed| =', np.abs(h_w - grid.scaled(t, 1.2).ptdf()).max())
    print(
        '  flow on T5-6 before / after:', round(float(h[t] @ grid.p), 2), '/', round(float(h_w[t] @ grid.p), 2), 'MW'
    )
    m2, det2 = modf_2x2(h, grid.frm, grid.to, *out)
    print('closed-form 2x2 MODF agrees with the solve:', np.abs(m2 - modf(h, grid.frm, grid.to, out)[0]).max(), det2)
    cut = [grid.names.index('L1-2'), grid.names.index('L1-5')]  # both branches of bus 1: a cut set
    print('cut set {L1-2, L1-5}: det =', modf_2x2(h, grid.frm, grid.to, *cut)[1])
    print(
        '  neither is a bridge: 1 - h_kk =',
        [round(float(1 - transfer(h, grid.frm, grid.to, [k])[k, 0]), 4) for k in cut],
    )
    print(
        '  column sums of H_OO (one MW across one cut branch crosses the cut in full):',
        transfer(h, grid.frm, grid.to, cut)[cut].sum(0),
    )


if __name__ == '__main__':
    main()
