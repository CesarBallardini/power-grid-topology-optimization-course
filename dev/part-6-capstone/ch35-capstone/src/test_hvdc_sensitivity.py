import hvdc_sensitivity
import numpy as np


def test_identity_holds_for_every_pair_including_the_slack():
    fr, to, b, p = hvdc_sensitivity.six_bus()
    for i in range(6):
        for j in range(6):
            if i != j:
                assert hvdc_sensitivity.check(fr, to, b, p, i, j) < 1e-12


def test_column_is_antisymmetric_and_zero_for_the_same_bus():
    fr, to, b, p = hvdc_sensitivity.six_bus()
    ptdf = hvdc_sensitivity.ptdf_matrix(fr, to, b, 6)
    np.testing.assert_allclose(hvdc_sensitivity.hvdc_column(ptdf, 1, 4), -hvdc_sensitivity.hvdc_column(ptdf, 4, 1))
    assert np.all(hvdc_sensitivity.hvdc_column(ptdf, 3, 3) == 0.0)


def test_total_transfer_is_conserved():
    # the setpoint leaves bus i and arrives at bus j: the net flow out of i changes by exactly -dP
    fr, to, b, p = hvdc_sensitivity.six_bus()
    ptdf = hvdc_sensitivity.ptdf_matrix(fr, to, b, 6)
    col = hvdc_sensitivity.hvdc_column(ptdf, 2, 5)
    out_of_2 = np.where(fr == 2, 1.0, 0.0) - np.where(to == 2, 1.0, 0.0)
    assert np.isclose(out_of_2 @ col, -1.0)
