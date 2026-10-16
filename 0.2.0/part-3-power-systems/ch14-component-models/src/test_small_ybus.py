import numpy as np
from small_ybus import RING_BC, pi_equivalent, ring3_ybus, row_sums, transformer_ybr


def test_ring_ybus_by_inspection():
    ybus = ring3_ybus()
    expected = np.array([[-13.98j, 10j, 4j], [10j, -14.98j, 5j], [4j, 5j, -8.98j]])
    assert np.allclose(ybus, expected)
    assert np.allclose(ybus, ybus.T)


def test_row_sums_equal_the_charging_at_each_bus():
    assert np.allclose(row_sums(ring3_ybus()), 1j * RING_BC)


def test_transformer_branch_matrix():
    ybr = transformer_ybr(1.05, 0.1)
    assert np.isclose(ybr[0, 0], -10j / 1.05**2)
    assert np.isclose(ybr[0, 1], 10j / 1.05) and np.isclose(ybr[1, 0], ybr[0, 1])
    assert np.isclose(ybr[1, 1], -10j)


def test_pi_equivalent_shunts_have_opposite_signs():
    series, shunt_from, shunt_to = pi_equivalent(transformer_ybr(1.05, 0.1))
    assert np.isclose(series, -10j / 1.05)
    assert shunt_from.imag > 0 > shunt_to.imag  # capacitive at the tapped end, inductive at the other
    assert np.isclose(shunt_from + shunt_to, -10j * (1 / 1.05 - 1) ** 2)  # ys (1/tau - 1)^2, nearly cancelling


def test_phase_shift_makes_ybr_asymmetric():
    ybr = transformer_ybr(1.0, 0.1, shift_deg=5.0)
    assert not np.isclose(ybr[0, 1], ybr[1, 0]) and np.isclose(abs(ybr[0, 1]), abs(ybr[1, 0]))
