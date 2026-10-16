import pytest
from big_m_study import instance, study, timed_solve


def test_instance_shapes_and_costs():
    f, c = instance(5, 12)
    assert len(f) == 5 and len(c) == 60
    assert all(3.0 <= v <= 8.0 for v in f.values())
    assert all(0.0 <= v <= 10.0 * 2**0.5 for v in c.values())


def test_both_formulations_agree_on_the_milp_and_the_strong_relaxation_is_tighter():
    f, c = instance(8, 20)
    milp_strong, _, nodes_strong = timed_solve(True, False, f, c)
    milp_big_m, _, nodes_big_m = timed_solve(False, False, f, c)
    assert milp_strong == pytest.approx(milp_big_m, rel=1e-6)
    lp_strong = timed_solve(True, True, f, c)[0]
    lp_big_m = timed_solve(False, True, f, c)[0]
    assert lp_big_m < lp_strong <= milp_strong + 1e-6
    assert nodes_strong >= 0 and nodes_big_m >= 0


def test_study_table_has_one_row_per_instance_and_formulation():
    table = study(((6, 15), (8, 20)))
    assert len(table) == 4
    assert (table.lp_relaxation <= table.milp + 1e-6).all()
    gaps = table['gap_%'].unstack('formulation')
    assert (gaps['big-M'] >= gaps['strong']).all()
