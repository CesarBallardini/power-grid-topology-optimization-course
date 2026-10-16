import numpy as np
import pytest
from kron_reduction import KEEP, kept_voltages_full, kept_voltages_reduced, kron_reduce
from ybus_numpy import build_ybus, case14_ppc


@pytest.fixture(scope='module')
def ybus():
    ppc = case14_ppc()
    return build_ybus(ppc['bus'], ppc['branch'], ppc['baseMVA'])[0]


def test_reduced_matrix_is_square_and_symmetric(ybus):
    reduced = kron_reduce(ybus, KEEP)
    assert reduced.shape == (5, 5)
    assert np.allclose(reduced, reduced.T)


def test_fill_in_appears_only_between_neighbours_of_eliminated_buses(ybus):
    # of the kept buses 1-5 only buses 4 and 5 touch the eliminated set (transformers 4-7, 4-9, 5-6)
    reduced = kron_reduce(ybus, KEEP)
    block = ybus[np.ix_(KEEP, KEEP)]
    changed = ~np.isclose(reduced, block)
    assert changed.tolist() == [
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, False, False],
        [False, False, False, True, True],
        [False, False, False, True, True],
    ]


def test_reduced_matrix_reproduces_kept_voltages(ybus):
    injection = np.array([1.0, -0.5j, 0.2 + 0.1j, -0.4, 0.3j])
    assert np.allclose(kept_voltages_full(ybus, KEEP, injection), kept_voltages_reduced(ybus, KEEP, injection))


def test_keeping_every_bus_changes_nothing(ybus):
    assert np.allclose(kron_reduce(ybus, np.arange(14)), ybus)


def test_two_bus_example_by_hand():
    # a chain 0 - 1 - 2 with unit admittances; eliminating the middle bus leaves a series admittance of 1/2
    chain = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=complex)
    reduced = kron_reduce(chain, np.array([0, 2]))
    assert np.allclose(reduced, [[0.5, -0.5], [-0.5, 0.5]])
