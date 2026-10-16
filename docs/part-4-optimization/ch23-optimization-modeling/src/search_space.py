"""How many topologies ToOp's exhaustive baseline would have to evaluate (Chapter 23, Exercise 4).

``count_workset_size`` mirrors the function of the same name in ToOp's ``dc_bruteforce/generator.py``: a sum over the
numbers of simultaneously split substations of the products of their action counts, times the number of ways to
switch off up to ``max_disconnections`` of the disconnectable branches.
"""

from itertools import combinations
from math import comb

import numpy as np

N_SUBS, N_DISCONNECTABLE, MAX_DISCONNECTIONS, EVALUATIONS_PER_SECOND = 40, 200, 3, 1e5


def action_counts(seed: int = 0, n_subs: int = N_SUBS) -> np.ndarray:
    """Number of valid split actions of each relevant substation, as the exercise draws them."""
    return np.random.default_rng(seed).integers(3, 61, size=n_subs)


def count_workset_size(groups: list[int], max_splits: int, n_disconnectable: int, max_disconnections: int) -> int:
    """Exact number of candidates: k-subset products summed over k <= max_splits, times the disconnection factor."""
    splits = min(max_splits, len(groups))
    disconnection_factor = sum(comb(n_disconnectable, d) for d in range(min(max_disconnections, n_disconnectable) + 1))
    total = 0
    for k in range(splits + 1):
        for chosen in combinations(groups, k):
            product = 1
            for n in chosen:
                product *= int(n)
            total += product
    return total * disconnection_factor


def seconds_to_text(seconds: float) -> str:
    """Seconds as the largest unit that reads well."""
    for unit, size in (('years', 365.25 * 86400), ('days', 86400), ('hours', 3600), ('minutes', 60)):
        if seconds >= size:
            return f'{seconds / size:,.1f} {unit}'
    return f'{seconds:,.1f} seconds'


def main() -> None:
    """Candidates and enumeration time for two and for four simultaneous splits."""
    counts = action_counts()
    print(f'action counts of the {N_SUBS} substations: {counts.tolist()} (sum {counts.sum()})')
    factor = sum(comb(N_DISCONNECTABLE, d) for d in range(MAX_DISCONNECTIONS + 1))
    print(
        f'disconnection factor: {factor:,} ways to switch off up to {MAX_DISCONNECTIONS} of {N_DISCONNECTABLE} lines'
    )
    for max_splits in (2, 4):
        n = count_workset_size(list(counts), max_splits, N_DISCONNECTABLE, MAX_DISCONNECTIONS)
        print(f'up to {max_splits} splits: {n:.3e} candidates, {seconds_to_text(n / EVALUATIONS_PER_SECOND)} at 1e5/s')


if __name__ == '__main__':
    main()
