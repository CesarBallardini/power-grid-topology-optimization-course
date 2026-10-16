import numpy as np
from ac_transfer import dc_estimate, dc_limit_overestimate, error_table, phasor_current, transfer_power


def test_worked_example_formula_matches_complex_power():
    v1, v2, x, delta = 1.05, 0.95, 0.1, 25.0
    p, q = transfer_power(v1, v2, x, delta)
    u1 = v1 * np.exp(1j * np.radians(delta))
    s12 = u1 * np.conj(phasor_current(v1, v2, x, delta))
    np.testing.assert_allclose([p, q], [s12.real, s12.imag], atol=1e-12)


def test_maximum_transfer_at_90_degrees():
    p, _ = transfer_power(1.0, 1.0, 0.1, np.arange(0, 181, 1.0))
    assert np.argmax(p) == 90
    assert p.max() == 10.0


def test_current_is_perpendicular_to_voltage_difference():
    for delta in (5.0, 30.0, 80.0):
        i = phasor_current(1.0, 1.0, 0.1, delta)
        u_diff = np.exp(1j * np.radians(delta)) - 1.0
        assert abs(np.degrees(np.angle(i / u_diff)) + 90.0) < 1e-9


def test_error_table_checkpoints():
    table = error_table()
    assert list(table.delta_deg) == list(range(0, 91, 5))
    by_angle = table.set_index('delta_deg').rel_error_pct
    np.testing.assert_allclose([by_angle[10], by_angle[30], by_angle[60]], [0.51, 4.72, 20.92], atol=0.005)
    np.testing.assert_allclose(table.p_dc_pu, dc_estimate(0.1, table.delta_deg.to_numpy()))
    assert (table.magnitude_error_pct == 0).all()


def test_error_factors_multiply():
    table = error_table(1.05, 0.95).set_index('delta_deg')
    total = 1 + table.rel_error_pct / 100
    product = (1 + table.angle_error_pct / 100) * (1 + table.magnitude_error_pct / 100)
    np.testing.assert_allclose(total, product, atol=1e-12)
    np.testing.assert_allclose(table.magnitude_error_pct.iloc[0], 100 * (1 / 0.9975 - 1))


def test_reactive_power_at_zero_angle_follows_magnitude():
    _, q_high_first = transfer_power(1.05, 1.0, 0.1, 0.0)
    _, q_low_first = transfer_power(1.0, 1.05, 0.1, 0.0)
    assert q_high_first > 0 > q_low_first
    _, q_flat = transfer_power(1.0, 1.0, 0.1, 10.0)
    np.testing.assert_allclose(q_flat, np.radians(10.0) ** 2 / (2 * 0.1), rtol=0.01)  # second order in delta


def test_dc_limit_overestimate():
    np.testing.assert_allclose(dc_limit_overestimate(0.95), 0.0526, atol=5e-5)
    assert dc_limit_overestimate(1.0) == 0.0
