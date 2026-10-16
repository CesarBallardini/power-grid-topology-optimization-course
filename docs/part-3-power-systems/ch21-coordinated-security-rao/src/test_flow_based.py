import numpy as np
from flow_based import BRANCHES, F_MAX, FRM, GLSK, P_REF, flow_based_domain, nodal_ptdf


def test_worked_example_numbers():
    ptdf_ab, ram, exchange = flow_based_domain(nodal_ptdf(BRANCHES, 4), GLSK, P_REF, F_MAX, FRM)
    np.testing.assert_allclose(ptdf_ab, [0.188, 0.412, 0.362, 0.138, 0.225], atol=5e-4)
    np.testing.assert_allclose(ram, [99.4, 130.6, 218.1, 186.9, 221.2], atol=0.05)
    np.testing.assert_allclose(np.round(exchange), [530, 317, 602, 1359, 983])


def test_zonal_ptdf_of_a_balanced_exchange_is_a_flow():
    ptdf = nodal_ptdf(BRANCHES, 4)
    ptdf_ab = flow_based_domain(ptdf, GLSK, P_REF, F_MAX, FRM)[0]
    # 100 MW exported by zone A (keys 0.6/0.4) and imported by zone B (0.5/0.5), computed nodally
    p = 100 * (GLSK[:, 0] - GLSK[:, 1])
    np.testing.assert_allclose(ptdf @ p, 100 * ptdf_ab)
