import numpy as np
import pytest
from domain_2d import CNECS, exchange_a_to_b, vertices, with_ram


def test_cnec_5_binds_at_150_mw():
    assert exchange_a_to_b(CNECS) == (pytest.approx(150.0), 'CNEC 5')


def test_slider_moves_the_binding_cnec():
    assert exchange_a_to_b(with_ram(CNECS, 'CNEC 5', 200.0)) == (pytest.approx(200.0), 'CNEC 5')
    assert exchange_a_to_b(with_ram(CNECS, 'CNEC 5', 300.0)) == (pytest.approx(275.0), 'CNEC 4')
    assert exchange_a_to_b(with_ram(CNECS, 'CNEC 5', 400.0)) == (pytest.approx(275.0), 'CNEC 4')


def test_domain_is_a_bounded_polygon_containing_the_origin():
    v = vertices(CNECS)
    assert len(v) == 5
    a = CNECS[['ptdf_A', 'ptdf_B']].to_numpy()
    assert np.all(a @ np.zeros(2) <= CNECS.ram.to_numpy())
    for corner in v:
        assert np.all(a @ corner <= CNECS.ram.to_numpy() + 1e-6)
