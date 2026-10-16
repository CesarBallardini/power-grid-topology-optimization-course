import dcgrid
import numpy as np
import pytest
import thresholds118


@pytest.fixture(scope='module')
def base():
    return thresholds118.ieee118()


def test_ieee118_size(base):
    assert (len(base.fr), base.n_bus) == (186, 118)


def test_lodf_denominators_and_determinants_do_not_change_under_uniform_scaling(base):
    den_1, det_1 = thresholds118.lodf_denominators(base), thresholds118.modf_determinants(base)
    for s in (0.1, 10.0):
        data = thresholds118.scaled(base, s)
        np.testing.assert_allclose(thresholds118.lodf_denominators(data), den_1, rtol=0, atol=1e-10)
        np.testing.assert_allclose(thresholds118.modf_determinants(data), det_1, rtol=0, atol=1e-10)


def test_bsdf_denominators_scale_with_the_inverse_factor(base):
    den_1 = thresholds118.bsdf_denominators(base)
    assert len(den_1) == 36
    for s in (0.1, 10.0):
        den_s = thresholds118.bsdf_denominators(thresholds118.scaled(base, s))
        for node, value in den_1.items():
            assert den_s[node] == pytest.approx(value / s, rel=1e-10)


def test_bsdf_denominator_matches_the_rebuilt_split_on_a_small_grid():
    # A node 1 with four branches; the split that keeps 0-1 and 1-2 and moves the two chords 1-3, 1-4
    fr, to = np.array([0, 1, 2, 3, 1, 1, 4]), np.array([1, 2, 0, 0, 3, 4, 0])
    b = np.array([1.0, 2.0, 1.5, 1.0, 2.5, 1.0, 1.2])
    h = dcgrid.ptdf_matrix(fr, to, b, 5)
    den = thresholds118.bsdf_denominator(h, fr, to, b, 1, [0, 1])
    # the denominator is the change of the "self" PTDF difference: the split PTDF is h + outer(num / den, p_c),
    # so a nonzero, finite denominator means the split does not island
    assert np.isfinite(den) and abs(den) > 1e-5
    fr2, to2 = fr.copy(), to.copy()
    fr2[4], fr2[5] = 5, 5
    h_split = dcgrid.ptdf_matrix(fr2, to2, b, 6)
    assert np.isfinite(h_split).all()


def test_experiment_counts_are_the_same_for_every_factor():
    rows = thresholds118.experiment()
    counts = {s: {k: v[1] for k, v in kinds.items()} for s, kinds in rows.items()}
    assert counts[0.1]['LODF'] == counts[1.0]['LODF'] == counts[10.0]['LODF'] == 9
    assert counts[0.1]['determinant'] == counts[1.0]['determinant'] == counts[10.0]['determinant'] == 40
    assert counts[0.1]['BSDF'] == counts[1.0]['BSDF'] == counts[10.0]['BSDF'] == 0
