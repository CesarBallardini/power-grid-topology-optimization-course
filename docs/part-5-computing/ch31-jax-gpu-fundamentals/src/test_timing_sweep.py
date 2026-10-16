import jax
from timing_sweep import agreement, repeated_sweep, sweep


def test_sweep_rows_for_both_precisions():
    rows = sweep(('case14',), repeats=2)
    assert [r['dtype'] for r in rows] == ['float64', 'float32']
    assert all(r['n_branch'] == 20 and r['first_s'] > 0 and r['steady_s'] > 0 for r in rows)
    assert rows[0]['output_mb'] == 20 * 20 * 8 / 1e6 and rows[1]['output_mb'] == 20 * 20 * 4 / 1e6
    assert all(r['first_s'] > r['steady_s'] for r in rows)  # the first call carries the compilation


def test_repeated_sweep_keeps_one_row_per_grid_and_precision():
    rows = repeated_sweep(('case14',), repeats=1, runs=2)
    assert [(r['case'], r['dtype']) for r in rows] == [('case14', 'float64'), ('case14', 'float32')]
    assert all(r['steady_s'] > 0 for r in rows)


def test_agreement_on_one_device_is_exact_and_float32_is_close():
    r = agreement('case14')
    exact, single = r['device_vs_cpu_float64'], r['float32_vs_float64']
    assert exact['flags_equal'] and exact['missed_bridges'] == 0
    if jax.default_backend() == 'cpu':
        assert exact['max_abs_mw'] == 0
    else:
        assert exact['max_rel'] < 1e-12
    assert single['max_rel'] < 1e-5
    assert single['missed_bridges'] <= 1  # case14 has one bridge; float32 rounding may hide it
