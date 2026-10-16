import numpy as np
import pytest
from resistor_lab import (
    FR,
    INJECTION,
    N_NODES,
    TO,
    G,
    build,
    checks,
    components,
    eigenvalue_figure,
    quadratic_form,
    random_network,
    rank_one_product,
    rank_one_update,
    solve,
    three_ways,
    time_three_ways,
    triangle,
    without_branch,
)


def test_full_network_laplacian_is_connected():
    _, gmat = build(FR, TO, G, N_NODES)
    c = checks(gmat)
    assert c['symmetric'] and c['max_row_sum'] == 0 and c['zero_eigenvalues'] == 1
    assert gmat[0, 0] == 15 and gmat[0, 1] == -10 and gmat[5, 5] == 2


def test_cutting_branch_4_5_isolates_node_5():
    _, g_cut = build(*without_branch(FR, TO, G, 6), N_NODES)
    assert checks(g_cut)['zero_eigenvalues'] == 2
    basis, comps = components(g_cut)
    assert basis.shape == (6, 2)
    assert comps == [[0, 1, 2, 3, 4], [5]]
    np.testing.assert_allclose(g_cut @ basis, 0, atol=1e-12)


def test_eigenvalue_figure_has_two_curves():
    _, gmat = build(FR, TO, G, N_NODES)
    _, g_cut = build(*without_branch(FR, TO, G, 6), N_NODES)
    fig = eigenvalue_figure(np.linalg.eigvalsh(gmat), np.linalg.eigvalsh(g_cut))
    assert len(fig.axes[0].lines) == 2


def test_nodal_solve_and_kcl_from_either_ground():
    a, _ = build(FR, TO, G, N_NODES)
    v0, cur0, res0 = solve(a, G, INJECTION, ground=0)
    v5, cur5, res5 = solve(a, G, INJECTION, ground=5)
    assert res0 < 1e-12 and res5 < 1e-12
    assert cur0[6] == pytest.approx(1.0)  # branch (4, 5) is the only way into node 5
    np.testing.assert_allclose(cur0, cur5, atol=1e-12)
    np.testing.assert_allclose(v5 - v0, v5[0], atol=1e-12)  # a constant shift: the null-space direction
    assert v5[0] == pytest.approx(0.7119, abs=5e-5)


def test_rank_one_update_matches_a_rebuilt_laplacian():
    _, gmat = build(FR, TO, G, N_NODES)
    a_new = np.array([1.0, 0, 0, 0, 0, -1.0])
    _, rebuilt = build(np.append(FR, 0), np.append(TO, 5), np.append(G, 1.0), N_NODES)
    np.testing.assert_allclose(rank_one_update(gmat, a_new, 1.0), rebuilt)
    x = np.random.default_rng(3).normal(size=N_NODES)
    lhs, rhs = rank_one_product(gmat, a_new, 1.0, x)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_triangle_null_space_is_the_constant_vector_with_or_without_a_branch():
    t = triangle()
    np.testing.assert_array_equal(t['AtA'], [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    assert t['rank'] == 2 and t['rank_without_branch'] == 2
    for null in (t['null'], t['null_without_branch']):
        assert null.shape == (3, 1)
        np.testing.assert_allclose(np.abs(null.ravel()), 1 / np.sqrt(3))


def test_quadratic_form_identity():
    _, gmat = build(FR, TO, G, N_NODES)
    x = np.random.default_rng(1).normal(size=N_NODES)
    lhs, rhs = quadratic_form(gmat, FR, TO, G, x)
    assert lhs == pytest.approx(rhs) and lhs > 0


def test_three_ways_agree_and_timing_runs_on_a_small_network():
    fr_, to_, g_ = random_network(50, 30)
    a, _ = build(fr_, to_, g_, 50)
    g1, g2, g3 = three_ways(a, g_)
    np.testing.assert_allclose(g1, g2)
    np.testing.assert_allclose(g2, g3)
    times = time_three_ways(n=200, extra=100)
    assert set(times) == {'np.diag', 'broadcast', 'einsum', 'einsum optimize=True'}
    assert all(t > 0 for t in times.values())
