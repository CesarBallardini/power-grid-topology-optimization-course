import numpy as np
from nodal import FR, INJ, TO, G, N
from wheatstone import bridge_currents, is_balanced, tellegen_check


def test_balanced_bridge_has_no_middle_current():
    r = [100.0, 200.0, 50.0, 100.0, 1000.0]  # R1 R4 = R2 R3 = 10 000
    assert is_balanced(r)
    v, i = bridge_currents(r)
    assert abs(v[2] - v[3]) < 1e-12 and abs(i[4]) < 1e-12
    assert abs(i[0] + i[1] - 1.0) < 1e-12  # the source current splits over R1 and R2


def test_unbalanced_bridge_has_middle_current():
    r = [100.0, 200.0, 50.0, 120.0, 1000.0]
    assert not is_balanced(r)
    v, i = bridge_currents(r)
    assert abs(i[4]) > 1e-4 and abs(i[4] - (v[2] - v[3]) / 1000.0) < 1e-12


def test_tellegen_with_currents_of_another_network():
    lhs, rhs = tellegen_check(FR, TO, G, INJ, N, [1.0] * len(G))
    assert abs(lhs - rhs) < 1e-12
    # the same with random conductances
    rng = np.random.default_rng(8)
    lhs, rhs = tellegen_check(FR, TO, G, INJ, N, rng.uniform(0.5, 5, len(G)))
    assert abs(lhs - rhs) < 1e-12
