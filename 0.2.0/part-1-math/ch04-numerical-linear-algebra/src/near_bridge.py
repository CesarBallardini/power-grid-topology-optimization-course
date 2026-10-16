"""A near-bridge in float32 and float64, and the condition number that explains it (Chapter 4, Lab 4, Exercise 6).

A weak branch (5, 0) of conductance g_w is added to the six-node network, so that (4, 5) is no longer a bridge
but almost one. Removing (4, 5) then has a Sherman–Morrison denominator of order g_w / 2, and the whole update
is evaluated in float32 and in float64 to watch round-off grow as the denominator shrinks.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sherman_morrison import fr, g, n, reduced_incidence, to

WEAK_CONDUCTANCES = (1e-2, 1e-4, 1e-6)
REMOVED = 6  # branch (4, 5), a bridge until the weak branch is added
NODE = 5  # node whose voltage is reported (reduced index 4)


def with_weak_branch(g_weak: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Branch lists with the extra branch (5, 0) of conductance ``g_weak`` appended."""
    return np.r_[fr, 5], np.r_[to, 0], np.r_[g, g_weak]


def removal_in_dtype(g_weak: float, dtype: type) -> tuple[float, float]:
    """Denominator and voltage at node 5 after removing (4, 5), with every array in ``dtype``.

    Matrix, inverse, incidence row and injection are all cast to ``dtype`` before any arithmetic, so the result is
    what a solver running entirely in that precision would produce.
    """
    fr_, to_, g_ = with_weak_branch(g_weak)
    ar = reduced_incidence(fr_, to_, n).astype(dtype)
    g_ = g_.astype(dtype)
    gr = ar.T @ (g_[:, None] * ar)
    minv = np.linalg.inv(gr)
    i_r = np.zeros(n - 1, dtype=dtype)
    i_r[NODE - 1] = 1
    v = minv @ i_r
    a = ar[REMOVED]
    u = minv @ a
    den = dtype(1) - g_[REMOVED] * (a @ u)
    v_after = v + u * (g_[REMOVED] * (a @ v) / den)
    return float(den), float(v_after[NODE - 1])


def resolved_float64(g_weak: float) -> float:
    """Voltage at node 5 after removing (4, 5), from a fresh float64 solve of the reduced network."""
    fr_, to_, g_ = with_weak_branch(g_weak)
    keep = np.arange(len(g_)) != REMOVED
    ar = reduced_incidence(fr_[keep], to_[keep], n)
    i_r = np.zeros(n - 1)
    i_r[NODE - 1] = 1
    return float(np.linalg.solve(ar.T @ np.diag(g_[keep]) @ ar, i_r)[NODE - 1])


def precision_table(weak: tuple[float, ...] = WEAK_CONDUCTANCES) -> pd.DataFrame:
    """Denominators, voltages and relative errors of both precisions against the re-solved answer (lab step 4)."""
    rows = []
    for g_weak in weak:
        exact = resolved_float64(g_weak)
        den32, v32 = removal_in_dtype(g_weak, np.float32)
        den64, v64 = removal_in_dtype(g_weak, np.float64)
        rows.append(
            {
                'g_weak': g_weak,
                'denominator': den64,
                'v5_resolved': exact,
                'v5_float32': v32,
                'v5_float64': v64,
                'rel_err_float32': abs(v32 - exact) / exact,
                'rel_err_float64': abs(v64 - exact) / exact,
            }
        )
    return pd.DataFrame(rows)


def condition_numbers(weak: np.ndarray, removed: bool = False) -> np.ndarray:
    """2-norm condition number of the reduced conductance matrix for each weak conductance (Exercise 6).

    With ``removed=False`` the matrix still holds branch (4, 5), as the exercise states; with ``removed=True`` it is
    the matrix after the removal, whose smallest eigenvalue is of the order of the weak conductance.
    """
    out = []
    for g_weak in weak:
        fr_, to_, g_ = with_weak_branch(g_weak)
        keep = np.arange(len(g_)) != REMOVED if removed else np.ones(len(g_), dtype=bool)
        ar = reduced_incidence(fr_[keep], to_[keep], n)
        out.append(np.linalg.cond(ar.T @ np.diag(g_[keep]) @ ar))
    return np.array(out)


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of the least-squares line through (log x, log y)."""
    return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])


def plot_condition(
    weak: np.ndarray, cond_before: np.ndarray, cond_after: np.ndarray, err32: np.ndarray
) -> matplotlib.figure.Figure:
    """Condition numbers before and after the removal and the float32 error, against g_w, log–log (Exercise 6)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(weak, cond_before, 'o-', label=r'$\kappa_2(G_r)$, branch (4, 5) present')
    ax.loglog(weak, cond_after, 'o-', label=r"$\kappa_2(G_r')$, branch (4, 5) removed")
    ax.loglog(weak, err32, 's--', label='float32 relative error of $v_5$')
    ax.set_xlabel('weak conductance $g_w$ (S)')
    ax.legend()
    return fig


def main() -> None:
    """Print the precision table of lab step 4 and the condition numbers of Exercise 6."""
    pd.set_option('display.width', 160)
    table = precision_table()
    print(table.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    weak = np.logspace(-6, -2, 9)
    before, after = condition_numbers(weak), condition_numbers(weak, removed=True)
    err32 = precision_table(tuple(weak))['rel_err_float32'].to_numpy()
    print("g_w        cond(Gr)   cond(Gr')   float32 rel err")
    for g_w, cb, ca, e in zip(weak, before, after, err32, strict=True):
        print(f'{g_w:.2e}  {cb:8.4g}  {ca:10.4g}  {e:.3g}')
    print(f'slope of log cond(Gr) vs log g_w  : {loglog_slope(weak, before):.3f}')
    print(f"slope of log cond(Gr') vs log g_w : {loglog_slope(weak, after):.3f}")
    print(f'slope of log float32 error vs log g_w: {loglog_slope(weak, err32):.3f}')
    ar0 = reduced_incidence(fr, to, n)
    print(f'cond without the weak branch (bridge intact): {np.linalg.cond(ar0.T @ np.diag(g) @ ar0):.4g}')


if __name__ == '__main__':
    main()
