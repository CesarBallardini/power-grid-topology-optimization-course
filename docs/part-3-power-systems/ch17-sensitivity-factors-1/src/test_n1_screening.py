import numpy as np
import pytest
from lodf import ptdf_matrix
from n1_screening import (
    bridges,
    brute_force_n1,
    case_data,
    distributed_slack_ptdf,
    double_outage,
    generator_outage,
    lodf_matrix,
    n1_matrix,
    pandapower_lodf,
    psdf_matrix,
    ptdf_after_outage,
    transfer_factors,
)


@pytest.fixture(scope='module')
def ieee14():
    frm, to, x, p, slack = case_data('case14')
    return frm, to, x, p, slack, ptdf_matrix(frm, to, x, len(p), slack)


def test_broadcast_n1_matches_brute_force(ieee14):
    frm, to, x, p, slack, h = ieee14
    n1 = n1_matrix(lodf_matrix(h, frm, to), h @ p)
    brute = brute_force_n1(frm, to, x, p, slack)
    ok = ~np.isnan(brute).any(axis=1)
    assert ok.sum() == len(frm) - 1  # IEEE 14 has exactly one bridge
    np.testing.assert_allclose(n1[ok], brute[ok], atol=1e-8)


def test_bridge_has_h_kk_one_and_nan_lodf_column(ieee14):
    frm, to, _, _, _, h = ieee14
    br = bridges(frm, to, h.shape[1])
    assert br == [18] and {frm[18], to[18]} == {6, 7}  # branch 7-8 of IEEE 14, pandapower's order (lines first)
    hkk = np.diag(transfer_factors(h, frm, to))
    assert hkk[18] == pytest.approx(1.0, abs=1e-12)
    assert (hkk[np.arange(20) != 18] < 1 - 1e-6).all()
    lodf = lodf_matrix(h, frm, to)
    assert np.isnan(lodf[:, 18]).all() and not np.isnan(np.delete(lodf, 18, axis=1)).any()
    pp_lodf = pandapower_lodf('case14')
    assert not np.isfinite(pp_lodf[:, 18]).all()
    np.testing.assert_allclose(np.delete(pp_lodf, 18, axis=1), np.delete(lodf, 18, axis=1), atol=1e-12)


def test_psdf_predicts_a_shift_as_equivalent_injections(ieee14):
    frm, to, x, p, slack, h = ieee14
    psdf = psdf_matrix(h, frm, to, x)
    k, phi = 7, np.deg2rad(3.0)  # a transformer
    p_eq = p.copy()
    p_eq[frm[k]] -= phi / x[k] * 100  # MW: injections in MW, b phi in pu
    p_eq[to[k]] += phi / x[k] * 100
    expected = h @ p_eq
    expected[k] += phi / x[k] * 100
    np.testing.assert_allclose(h @ p + psdf[:, k] * phi * 100, expected, atol=1e-9)


def test_distributed_slack_changes_ptdf_but_not_flows_or_lodf(ieee14):
    frm, to, _, p, _, h = ieee14
    w = np.full(h.shape[1], 1 / h.shape[1])
    hw = distributed_slack_ptdf(h, w)
    assert np.abs(hw - h).max() > 0.01
    np.testing.assert_allclose(hw @ p, h @ p, atol=1e-9)
    ok = np.arange(20) != 18
    np.testing.assert_allclose(lodf_matrix(hw, frm, to)[:, ok], lodf_matrix(h, frm, to)[:, ok], atol=1e-9)
    assert np.abs(generator_outage(h, p, 1, 40.0) - generator_outage(h, p, 1, 40.0, w)).max() > 1.0


def test_ptdf_update_equals_recomputation(ieee14):
    frm, to, x, _, slack, h = ieee14
    for k in (0, 5, 16):
        keep = np.arange(20) != k
        h_ref = np.zeros_like(h)
        h_ref[keep] = ptdf_matrix(frm[keep], to[keep], x[keep], 14, slack)
        np.testing.assert_allclose(ptdf_after_outage(h, frm, to, k), h_ref, atol=1e-9)


def test_double_outage_by_two_lodf_steps(ieee14):
    frm, to, x, p, slack, h = ieee14
    k1, k2 = 2, 9
    keep = (np.arange(20) != k1) & (np.arange(20) != k2)
    f_ref = np.zeros(20)
    f_ref[keep] = ptdf_matrix(frm[keep], to[keep], x[keep], 14, slack) @ p
    np.testing.assert_allclose(double_outage(h, frm, to, h @ p, k1, k2), f_ref, atol=1e-8)
    np.testing.assert_allclose(double_outage(h, frm, to, h @ p, k2, k1), f_ref, atol=1e-8)  # order does not matter
