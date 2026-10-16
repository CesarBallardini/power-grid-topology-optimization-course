import bsdf_property
import numpy as np
from hypothesis import given, settings


@settings(max_examples=150, deadline=None)
@given(bsdf_property.split_grids())
def test_bsdf_update_equals_rebuilt_ptdf(grid):
    mismatch, _ = bsdf_property.violation(grid)
    assert mismatch < 1e-8


@settings(max_examples=150, deadline=None)
@given(bsdf_property.split_grids())
def test_kcl_holds_at_both_busbars(grid):
    _, residual = bsdf_property.violation(grid)
    assert residual < 1e-8


def test_off_by_one_has_a_counterexample_with_three_buses():
    fr, to, b, p, node, to_b = bsdf_property.smallest_counterexample()
    assert len(p) == 3
    assert len(bsdf_property.branches_at(fr, to, node)) == 4
    assert bsdf_property.violation((fr, to, b, p, node, to_b), off_by_one=True)[0] > 1e-6
    assert bsdf_property.violation((fr, to, b, p, node, to_b))[0] < 1e-8


def test_kcl_residuals_are_zero_on_a_hand_made_split():
    fr, to = np.array([0, 1, 2, 0, 0]), np.array([1, 2, 0, 1, 1])
    b, p = np.ones(5), np.array([0.3, 0.5, -0.8])
    h = bsdf_property.dcgrid.ptdf_matrix(fr, to, b, 3)
    to_b = np.array([False, True, True, True])
    h_split = bsdf_property.split_ptdf(h, fr, to, b, 1, to_b)
    assert np.abs(bsdf_property.kcl_residuals(h_split, fr, to, p, 1, to_b)).max() < 1e-12
