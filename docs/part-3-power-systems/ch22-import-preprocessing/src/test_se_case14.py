import numpy as np
from se_case14 import (
    add_measurements,
    chi2,
    corrupt_first_line_flow,
    estimate,
    estimation_errors,
    normalized_residuals,
    remove_bad_data,
    solved_case14,
)


def test_exact_measurements_reproduce_the_power_flow():
    net = solved_case14()
    add_measurements(net, None)
    assert estimate(net)
    dv, da = estimation_errors(net)
    assert dv < 1e-8 and da < 1e-6
    assert not chi2(net)


def test_noisy_measurements_pass_and_a_bad_datum_is_found():
    net = solved_case14()
    add_measurements(net, np.random.default_rng(14))
    assert estimate(net)
    dv, da = estimation_errors(net)
    assert dv < 0.01 and da < 0.5
    assert not chi2(net)
    idx = corrupt_first_line_flow(net)
    assert chi2(net)
    estimate(net)
    assert normalized_residuals(net).index[0] == idx
    assert idx in remove_bad_data(net)


def test_topology_error_shows_up_around_the_opened_line():
    net = solved_case14()
    add_measurements(net, np.random.default_rng(14))
    net.line.loc[8, 'in_service'] = False
    assert estimate(net)
    assert chi2(net)
    top = normalized_residuals(net).head(6)
    ends = {int(net.line.from_bus[8]), int(net.line.to_bus[8])}
    touching = [
        (r.element_type == 'bus' and r.element in ends) or (r.element_type == 'line' and r.element == 8)
        for r in top.itertuples()
    ]
    assert sum(touching) >= 3
    assert top.r_n.abs().iloc[0] > 20
