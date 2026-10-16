import numpy as np
import pytest
from rx_plane import (
    apparent_impedance,
    boundary_distance,
    contains,
    infeed_apparent_impedance,
    is_simple,
    is_valid,
    quadrilateral_zone,
    toop_polygon,
    tripping_mva,
    trips,
)

POLY = toop_polygon(30.0, 2.5, 10.0, 15.0)


def test_polygon_vertices_and_validity():
    assert POLY[0] == (0.0, 0.0) and POLY[1] == (2.5, 0.0) and POLY[4] == (10.0, 15.0) and POLY[5] == (0.0, 15.0)
    assert POLY[2][1] == pytest.approx(2.5 * np.tan(np.radians(30))) and POLY[3][1] == pytest.approx(
        10 * np.tan(np.radians(30))
    )
    assert is_valid(30.0, 2.5, 10.0, 15.0) and is_simple(POLY)
    assert not is_valid(80.0, 2.5, 10.0, 15.0) and not is_simple(toop_polygon(80.0, 2.5, 10.0, 15.0))
    assert is_valid(56.3, 2.5, 10.0, 15.0) and not is_valid(56.4, 2.5, 10.0, 15.0)  # tan(56.31 deg) = 1.5


def test_point_in_polygon():
    assert contains(POLY, (5.0, 10.0)) and contains(POLY, (1.0, 1.0))
    assert not contains(POLY, (5.0, 1.0))  # below the sloped edge
    assert not contains(POLY, (11.0, 5.0)) and not contains(POLY, (5.0, 16.0))
    assert contains(POLY, (0.0, 7.0))  # on the left edge


def test_apparent_impedance_magnitude_and_angle():
    r, x = apparent_impedance(380.0, 950.0, 312.25)
    s = abs(complex(950.0, 312.25))
    assert np.hypot(r, x) == pytest.approx(380.0**2 / s)
    assert np.arctan2(x, r) == pytest.approx(np.arctan2(312.25, 950.0))


def test_tripping_mva_follows_the_boundary_and_the_factor():
    phi = 0.0
    zb = boundary_distance(POLY, phi)
    assert zb == pytest.approx(2.5)  # along the R axis the polygon ends at r_i
    assert tripping_mva(POLY, 380.0, 0.0) == pytest.approx(380.0**2 / 2.5)
    assert tripping_mva(POLY, 380.0, 0.0, 1.41) == pytest.approx(380.0**2 / (2.5 * 1.41))
    s = tripping_mva(POLY, 380.0, 18.2)
    just_above = apparent_impedance(380.0, 1.001 * s * np.cos(np.radians(18.2)), 1.001 * s * np.sin(np.radians(18.2)))
    just_below = apparent_impedance(380.0, 0.999 * s * np.cos(np.radians(18.2)), 0.999 * s * np.sin(np.radians(18.2)))
    assert trips(POLY, just_above) and not trips(POLY, just_below)


def test_absolute_values_make_the_test_non_directional():
    z = apparent_impedance(380.0, -60000.0 * 0.95, -60000.0 * np.sqrt(1 - 0.95**2))
    assert not contains(POLY, z) and trips(POLY, z)


def test_infeed_pushes_the_fault_out_of_zone_2():
    z_line = 100 * complex(0.03, 0.3)
    zone1, zone2 = quadrilateral_zone(0.8 * z_line.imag, 10.0), quadrilateral_zone(1.2 * z_line.imag, 10.0)
    z0 = infeed_apparent_impedance(z_line, 0.9, 5.0, 0.0)
    z1 = infeed_apparent_impedance(z_line, 0.9, 5.0, 0.5)
    assert z0 == pytest.approx(complex(7.7, 27.0)) and z1 == pytest.approx(complex(12.7, 27.0))
    assert not contains(zone1, (z0.real, z0.imag)) and contains(zone2, (z0.real, z0.imag))
    assert not contains(zone2, (z1.real, z1.imag))
    assert contains(zone1, (2.4, 24.0 * 0.5))  # a close-in bolted fault is in zone 1
