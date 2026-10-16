import numpy as np
from trafo_equivalents import flow_share_with_ratio, referred_impedance, star_equivalent, susceptance_change


def test_exercise_1_referred_impedance():
    np.testing.assert_allclose(referred_impedance(10.0, 380.0, 110.0), 119.34, atol=0.005)
    np.testing.assert_allclose(referred_impedance(referred_impedance(10.0, 380.0, 110.0), 110.0, 380.0), 10.0)


def test_exercise_2_ratio_raises_susceptance_and_flow():
    np.testing.assert_allclose(susceptance_change(1.05), 0.05)
    assert flow_share_with_ratio(1.05) > flow_share_with_ratio(1.0) == 0.75
    np.testing.assert_allclose(flow_share_with_ratio(1.05), (1.05 / 0.1) / (1.05 / 0.1 + 1 / 0.3))


def test_exercise_3_star_equivalent():
    x1, x2, x3 = star_equivalent(0.10, 0.25, 0.20)
    np.testing.assert_allclose([x1, x2, x3], [0.075, 0.025, 0.175])
    np.testing.assert_allclose([x1 + x2, x1 + x3, x2 + x3], [0.10, 0.25, 0.20])
