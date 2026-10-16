"""Removing two branches at once with the Woodbury identity and a closed-form 2×2 solve (Chapter 4, Exercise 3).

Removing branches j and k changes the reduced Laplacian by −U C Uᵀ with U = [aⱼ aₖ] and C = diag(gⱼ, gₖ), so

    (Gr − U C Uᵀ)⁻¹ = Gr⁻¹ + Gr⁻¹ U (C⁻¹ − Uᵀ Gr⁻¹ U)⁻¹ Uᵀ Gr⁻¹,

and the only new system to solve is 2×2. Its determinant is zero exactly when the pair of removals islands part
of the network, which is what ToOp's ``solve2x2`` (``dc_solver/jax/unrolled_linalg.py``) reports with its Boolean
flag.
"""

import numpy as np
from sherman_morrison import fr, g, n, reduced_incidence, to

DET_TOL = 1e-10  # the same threshold as ToOp's solve2x2


def solve2x2(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, bool]:
    """Solve a 2×2 system in closed form; the flag is False when |det| ≤ 1e-10 (the solution is then meaningless).

    Mirrors ``solve2x2`` in ToOp: adjugate over determinant, one or several right-hand sides, no exception on a
    singular matrix so that it can run inside a vectorized GPU kernel.
    """
    (m1, m2), (m3, m4) = a
    det = m1 * m4 - m2 * m3
    with np.errstate(divide='ignore', invalid='ignore'):
        a_inv = np.array([[m4, -m2], [-m3, m1]]) / det
        x = a_inv @ b
    return x, bool(abs(det) > DET_TOL)


def remove_two_branches(minv: np.ndarray, ar: np.ndarray, cond: np.ndarray, j: int, k: int) -> tuple[np.ndarray, bool]:
    """Inverse of the reduced Laplacian after removing branches ``j`` and ``k``, by Woodbury, and a validity flag."""
    u = ar[[j, k]].T  # (n-1) × 2
    w = minv @ u  # Gr⁻¹ U
    inner = np.diag(1 / cond[[j, k]]) - u.T @ w  # C⁻¹ − Uᵀ Gr⁻¹ U
    x, ok = solve2x2(inner, w.T)  # inner⁻¹ Uᵀ Gr⁻¹
    return minv + w @ x, ok


def fresh_inverse(ar: np.ndarray, cond: np.ndarray, removed: tuple[int, ...]) -> np.ndarray:
    """Inverse of the reduced Laplacian rebuilt without the ``removed`` branches."""
    keep = ~np.isin(np.arange(len(cond)), removed)
    return np.linalg.inv(ar[keep].T @ np.diag(cond[keep]) @ ar[keep])


def main() -> None:
    """Print the Woodbury check for branches (1, 2) and (3, 4), and the four pairs whose removal islands nodes."""
    ar = reduced_incidence(fr, to, n)
    minv = np.linalg.inv(ar.T @ np.diag(g) @ ar)
    for j, k in ((2, 5), (0, 1), (3, 4), (3, 5), (4, 5)):
        updated, ok = remove_two_branches(minv, ar, g, j, k)
        ends = f'({fr[j]}, {to[j]}) and ({fr[k]}, {to[k]})'
        if ok:
            gap = np.max(np.abs(updated - fresh_inverse(ar, g, (j, k))))
            print(f'remove {ends}: det ok, max |Woodbury - fresh| = {gap:.1e}')
        else:
            print(f'remove {ends}: det ~ 0, the pair islands part of the network')
    inner = np.diag(1 / g[[4, 5]]) - ar[[4, 5]] @ minv @ ar[[4, 5]].T
    print('2x2 matrix for (2, 4), (3, 4):', np.round(inner, 6).tolist(), ' det =', f'{np.linalg.det(inner):.2e}')


if __name__ == '__main__':
    main()
