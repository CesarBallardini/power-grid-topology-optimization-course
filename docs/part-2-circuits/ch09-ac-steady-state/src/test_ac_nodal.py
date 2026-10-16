import numpy as np
from ac_nodal import FR, I_SOURCE, TO, V_SOURCE, Z_BRANCH, admittance_matrix, power_balance, solve


def test_admittance_matrix_rows_sum_to_zero():
    y = admittance_matrix(FR, TO, Z_BRANCH, 3)
    np.testing.assert_allclose(y.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(y, y.T)


def test_kcl_and_power_balance():
    v, inj, i_branch = solve(FR, TO, Z_BRANCH, V_SOURCE, I_SOURCE)
    assert v[0] == 0 and v[1] == V_SOURCE
    np.testing.assert_allclose(inj[2], I_SOURCE, atol=1e-9)  # the current source injection is honoured
    np.testing.assert_allclose(inj.sum(), 0, atol=1e-9)  # the ground returns the rest
    np.testing.assert_allclose(i_branch[0], inj[1], atol=1e-9)  # the source current is the RL current
    injected, absorbed = power_balance(v, inj, i_branch, Z_BRANCH)
    np.testing.assert_allclose(injected, absorbed, atol=1e-9)
    assert absorbed.real > 0 and abs(absorbed.imag) > 0
