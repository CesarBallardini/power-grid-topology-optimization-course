"""Exercise 5 of Chapter 18: floating-point operation counts of ToOp's low-rank updates.

A multiply-add counts as two flops, a k x k LU solve as 2k³/3 (its leading term), and gathers and sign flips as
nothing. ``n_l`` branches, ``n_b`` nodes, ``d`` branches at the split station of which ``a`` stay on bus A, ``k``
simultaneous outages, ``n_f`` N-1 cases and ``t`` timesteps.
"""


def bsdf_flops(n_l: int, n_b: int, d: int, a: int) -> int:
    """One BSDF update: coupler row, numerator, denominator, beta and the rank-1 PTDF update."""
    coupler_row = d * n_b  # sum of d PTDF rows
    numerator = 2 * a * n_l  # a scaled differences of PTDF columns
    denominator = 4 * a  # two sums over bus-A branches
    beta = n_l  # one division per branch
    update = 2 * n_l * n_b  # H + beta p_c
    return coupler_row + numerator + denominator + beta + update


def modf_flops(n_l: int, n_b: int, k: int) -> int:
    """One k-outage MODF: the k x k system, the n_l x k MODF matrix and the rank-k PTDF update."""
    transfer = n_l * k  # script-H[:, O] = H[:, f] - H[:, t]
    solve = 2 * k**3 // 3  # LU of I - script-H[O, O]
    modf_matrix = 2 * n_l * k * k  # N (I - script-H[O, O])^{-1}
    update = 2 * n_l * n_b * k  # H + MODF H[O, :]
    return transfer + solve + modf_matrix + update


def lodf_matrix_flops(n_l: int, n_f: int) -> int:
    """The LODF matrix for n_f single outages: one difference and one division per entry, n_f denominators."""
    return 2 * n_l * n_f + n_f


def n1_flows_flops(n_l: int, n_b: int, n_f: int, t: int) -> int:
    """N-0 flows H P for t timesteps and the N-1 matrix F + LODF[:, k] F_k for n_f outages each."""
    return 2 * n_l * n_b * t + 2 * n_l * n_f * t


def table(
    n_l: int, n_b: int, d: int = 5, a: int = 2, k: int = 3, n_f: int | None = None, t: int = 1
) -> dict[str, int]:
    """Every count for one grid size; ``n_f`` defaults to all branches."""
    n_f = n_l if n_f is None else n_f
    return {
        'BSDF update (one split)': bsdf_flops(n_l, n_b, d, a),
        f'MODF update (k = {k})': modf_flops(n_l, n_b, k),
        f'LODF matrix (n_f = {n_f})': lodf_matrix_flops(n_l, n_f),
        f'N-1 flow matrix (T = {t})': n1_flows_flops(n_l, n_b, n_f, t),
        'full PTDF recomputation': 2 * n_b**3 // 3 + 2 * n_l * n_b * n_b,
    }


def main() -> None:
    """Counts for IEEE 14 and for a transmission grid of ToOp's size."""
    for label, n_l, n_b, t in (('IEEE 14', 20, 14, 1), ('a 1 000-node grid, one day', 1500, 1000, 24)):
        print(label)
        for name, flops in table(n_l, n_b, t=t).items():
            print(f'  {name:28s} {flops:14,d} flops')


if __name__ == '__main__':
    main()
