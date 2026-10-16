import numpy as np
from nodal import (
    EX_FR,
    EX_G,
    EX_INJ,
    EX_TO,
    FR,
    INJ,
    TO,
    G,
    N,
    effective_resistance,
    incidence,
    kcl_residual,
    nodal_solve,
    power_balance,
    split_node_2,
    superposition_check,
    thevenin_resistance,
)


def test_worked_example():
    v, i = nodal_solve(EX_FR, EX_TO, EX_G, EX_INJ, 3)
    np.testing.assert_allclose(v, [0, 18 / 14, 6 / 14])
    np.testing.assert_allclose(i, [18 / 14, 12 / 7, 12 / 7])


def test_incidence_rows_sum_to_zero():
    a = incidence(FR, TO, N)
    assert a.shape == (7, 6) and np.all(a.sum(axis=1) == 0)
    gmat = a.T @ np.diag(G) @ a
    np.testing.assert_allclose(gmat, gmat.T)
    np.testing.assert_allclose(gmat.sum(axis=1), 0, atol=1e-12)  # a weighted Laplacian


def test_lab_network_kcl_and_power_balance():
    v, i = nodal_solve(FR, TO, G, INJ, N)
    np.testing.assert_allclose(v, [0, -0.03041, -0.03918, -0.08889, -0.16686, -0.66686], atol=5e-6)
    np.testing.assert_allclose(kcl_residual(FR, TO, i, INJ, N), 0, atol=1e-12)  # INJ is balanced, ground included
    sources, branches = power_balance(v, i, INJ, G)
    assert abs(sources - branches) < 1e-12 and abs(sources - 0.647271) < 1e-6


def test_superposition():
    v_both, v_sum = superposition_check()
    np.testing.assert_allclose(v_both, v_sum, atol=1e-14)


def test_thevenin_equals_effective_resistance():
    rth = thevenin_resistance(FR, TO, G, N, 5, 0)
    assert abs(rth - 0.71189) < 5e-6
    assert abs(rth - effective_resistance(FR, TO, G, N, 5, 0)) < 1e-12
    # series path 4-5 (1/2 ohm) plus the rest: the Thevenin resistance exceeds 0.5 ohm
    assert rth > 0.5


def test_split_node_coupler_current_is_the_unsplit_imbalance():
    s = split_node_2()
    assert abs(s['imbalance_a'] - 0.76608) < 5e-6
    assert abs(s['coupler'] - s['imbalance_a']) < 1e-6
    weak = split_node_2(10.0)
    assert abs(weak['coupler'] - 0.68443) < 5e-6 and weak['coupler'] < s['coupler']
    mid = split_node_2(1e3)
    assert abs(mid['coupler'] - 0.76517) < 5e-6


def test_exercise_4_other_ground_shifts_voltages_keeps_currents():
    v0, i0 = nodal_solve(FR, TO, G, INJ, N, ground=0)
    v3, i3 = nodal_solve(FR, TO, G, INJ, N, ground=3)
    np.testing.assert_allclose(i3, i0, atol=1e-12)
    shift = v3 - v0
    np.testing.assert_allclose(shift, shift[0], atol=1e-12)
    assert abs(shift[0] - 0.08889) < 5e-6
