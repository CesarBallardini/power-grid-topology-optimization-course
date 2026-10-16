from itertools import combinations, product
from math import comb

import numpy as np
from search_space import action_counts, count_workset_size, seconds_to_text


def brute_force(groups, max_splits, n_disc, max_disc):
    topologies = set()
    for k in range(max_splits + 1):
        for subs in combinations(range(len(groups)), k):
            for actions in product(*(range(groups[s]) for s in subs)):
                for d in range(max_disc + 1):
                    for branches in combinations(range(n_disc), d):
                        topologies.add((subs, actions, branches))
    return len(topologies)


def test_matches_brute_force_on_a_small_case():
    groups = [3, 4, 2]
    assert count_workset_size(groups, 2, 5, 2) == brute_force(groups, 2, 5, 2)
    assert count_workset_size(groups, 5, 5, 1) == brute_force(groups, 3, 5, 1)  # max_splits is capped at len(groups)


def test_closed_form_for_two_splits():
    counts = action_counts()
    s1 = int(counts.sum())
    s2 = sum(int(a) * int(b) for a, b in combinations(counts, 2))
    factor = sum(comb(200, d) for d in range(4))
    assert count_workset_size(list(counts), 2, 200, 3) == (1 + s1 + s2) * factor
    assert len(counts) == 40 and counts.min() >= 3 and counts.max() <= 60
    assert np.array_equal(counts, np.random.default_rng(0).integers(3, 61, size=40))


def test_seconds_to_text():
    assert seconds_to_text(90) == '1.5 minutes'
    assert seconds_to_text(2 * 365.25 * 86400) == '2.0 years'
