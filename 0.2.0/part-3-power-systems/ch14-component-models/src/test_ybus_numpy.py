import numpy as np
import pytest
from pandapower.pypower.idx_brch import SHIFT, TAP
from ybus_numpy import (
    asymmetry,
    branch_admittances,
    build_ybus,
    case14_ppc,
    connection_matrices,
    max_difference,
    transformer_branches,
    with_shift,
)


@pytest.fixture(scope='module')
def ppc():
    return case14_ppc()


def test_connection_matrices_have_one_entry_per_row(ppc):
    cf, ct = connection_matrices(ppc['branch'], ppc['bus'].shape[0])
    assert cf.shape == ct.shape == (20, 14)
    assert (cf.sum(axis=1) == 1).all() and (ct.sum(axis=1) == 1).all()
    assert not (cf * ct).any()  # no branch starts and ends at the same bus


def test_ybus_matches_pandapower_to_machine_precision(ppc):
    assert max_difference(ppc) < 1e-12


def test_ybus_is_symmetric_without_phase_shift(ppc):
    ybus, _, _ = build_ybus(ppc['bus'], ppc['branch'], ppc['baseMVA'])
    assert asymmetry(ybus) < 1e-12
    assert ybus.shape == (14, 14) and np.count_nonzero(ybus) == 14 + 2 * 20


def test_three_transformers_with_off_nominal_taps(ppc):
    trafos = transformer_branches(ppc['branch'])
    assert len(trafos) == 3
    assert np.allclose(np.sort(ppc['branch'][trafos, TAP].real), [0.932, 0.969, 0.978])


def test_phase_shift_breaks_symmetry_but_not_the_modulus(ppc):
    k = int(transformer_branches(ppc['branch'])[0])
    shifted = with_shift(ppc['branch'], k, 5.0)
    assert shifted[k, SHIFT] == 5.0 and ppc['branch'][k, SHIFT] == 0.0
    yff, yft, ytf, ytt = branch_admittances(shifted)
    assert yft[k] != ytf[k] and np.isclose(abs(yft[k]), abs(ytf[k]))
    ybus, _, _ = build_ybus(ppc['bus'], shifted, ppc['baseMVA'])
    assert asymmetry(ybus) > 0.1
