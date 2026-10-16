from flops import bsdf_flops, lodf_matrix_flops, modf_flops, n1_flows_flops, table


def test_leading_terms():
    n_l, n_b = 1000, 500
    assert bsdf_flops(n_l, n_b, 5, 2) == 2 * n_l * n_b + 5 * n_b + 4 * n_l + 8 + n_l
    assert modf_flops(n_l, n_b, 1) == n_l + 0 + 2 * n_l + 2 * n_l * n_b
    assert lodf_matrix_flops(n_l, n_l) == 2 * n_l * n_l + n_l
    assert n1_flows_flops(n_l, n_b, n_l, 24) == 24 * (2 * n_l * n_b + 2 * n_l * n_l)


def test_modf_grows_linearly_in_k_for_small_k_and_updates_beat_recomputation():
    n_l, n_b = 1500, 1000
    t = table(n_l, n_b)
    assert modf_flops(n_l, n_b, 2) < 2.1 * modf_flops(n_l, n_b, 1)
    assert t['BSDF update (one split)'] < t['full PTDF recomputation'] / 100
    assert t['MODF update (k = 3)'] < t['full PTDF recomputation'] / 100


def test_ieee14_numbers_of_the_solution():
    t = table(20, 14)
    assert t['BSDF update (one split)'] == 738
    assert t['MODF update (k = 3)'] == 2118
    assert t['LODF matrix (n_f = 20)'] == 820
    assert t['N-1 flow matrix (T = 1)'] == 1360
