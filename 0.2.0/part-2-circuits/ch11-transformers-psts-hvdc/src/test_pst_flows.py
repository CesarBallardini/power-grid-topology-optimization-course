import numpy as np
from dc_nodal import incidence, nodal_solve
from pst_flows import (
    FR,
    INJ,
    TO,
    B,
    alpha_for_zero_flow_b,
    psdf_column,
    slope_per_degree,
    two_path_by_injections,
    two_path_flows,
)


def test_worked_example_numbers():
    f_a, f_b = two_path_flows(0.0)
    np.testing.assert_allclose([f_a, f_b], [0.75, 0.25])
    np.testing.assert_allclose(slope_per_degree(), 0.0436, atol=5e-5)
    f_a1, _ = two_path_flows(1.0)
    np.testing.assert_allclose(f_a1 - f_a, slope_per_degree())


def test_flow_is_affine_and_line_b_reverses():
    alpha = np.linspace(-10, 10, 21)
    f_a, f_b = two_path_flows(alpha)
    np.testing.assert_allclose(np.diff(f_a), slope_per_degree(), atol=1e-12)
    np.testing.assert_allclose(f_a + f_b, 1.0)
    a0 = alpha_for_zero_flow_b()
    np.testing.assert_allclose(a0, np.degrees(0.1), atol=1e-12)
    assert abs(two_path_flows(a0)[1]) < 1e-12
    assert two_path_flows(a0 + 1)[1] < 0 < two_path_flows(a0 - 1)[1]


def test_equivalent_injections_match_closed_formula():
    for alpha in (-10.0, -3.0, 0.0, 4.5, 10.0):
        np.testing.assert_allclose(two_path_by_injections(alpha), two_path_flows(alpha), atol=1e-12)


def test_psdf_column_is_exact_and_step_independent():
    col1 = psdf_column(FR, TO, B, 3, INJ, 6, step_deg=1.0)
    col2 = psdf_column(FR, TO, B, 3, INJ, 6, step_deg=0.01)
    np.testing.assert_allclose(col1, col2, atol=1e-9)
    # the sensitivities do not depend on the injections (linear model)
    col3 = psdf_column(FR, TO, B, 3, np.zeros(6), 6)
    np.testing.assert_allclose(col1, col3, atol=1e-12)
    # the changes are a circulation: zero net change at every node (loop flow only)
    a = incidence(FR, TO, 6)
    np.testing.assert_allclose(a.T @ col1, 0, atol=1e-12)
    assert col1[6] == 0  # branch (4, 5) feeds a leaf: its flow is fixed by KCL
    assert col1[3] > 0  # the PST pushes flow through its own branch
    _, base = nodal_solve(FR, TO, B, INJ, 6)
    np.testing.assert_allclose(base[6], 1.0)
