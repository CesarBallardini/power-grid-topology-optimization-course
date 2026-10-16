import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

from solver_lab_toop import (  # noqa: E402
    batch_sweep,
    compile_vs_run,
    compiled_programs,
    default_topology,
    grid_folder,
    passthrough,
    run_solver_symmetric,
    setup,
    trigger_study,
)


@pytest.fixture(scope='module')
def si():
    return setup(grid_folder('case57'))


def test_first_call_compiles_and_later_calls_hit_the_cache(si):
    rows = compile_vs_run(si)
    names = [r[0] for r in rows]
    by_name = dict(zip(names, rows, strict=True))
    assert by_name['run_solver, call 1'][2] and not by_name['run_solver, call 2'][2]
    assert by_name['run_solver, call 1'][1] > by_name['run_solver, call 3'][1]
    assert not by_name['run_solver_symmetric, passthrough defined once, call 2'][2]
    assert all(r[2] for r in rows if 'fresh lambda' in r[0])  # every fresh lambda traces again


def test_cache_entries_separate_the_two_aggregation_functions(si):
    sc, di = si.solver_config, si.dynamic_information
    topologies = default_topology(sc, batch_size=4)
    run_solver_symmetric(topologies, None, None, di, sc, passthrough)
    before = compiled_programs()
    run_solver_symmetric(topologies, None, None, di, sc, passthrough)
    assert compiled_programs() == before  # the module-level function hits the cache
    run_solver_symmetric(topologies, None, None, di, sc, lambda lf: (lf.n_0_matrix, lf.n_1_matrix))
    assert compiled_programs() == before + 1  # a fresh lambda is a new cache key


def test_trigger_study_matches_the_predictions(si):
    rows = {r['change']: r for r in trigger_study(si)}
    base = rows['baseline, second call']
    assert not base['retraced']
    assert rows['replace(sc), no field changed']['retraced']
    assert rows['replace(sc, number_most_affected=20)']['retraced']
    assert rows['replace(sc, number_most_affected=20)']['shape'][-1] == 20
    mutated = rows['same copy, sc.number_most_affected = 20 in place']
    assert not mutated['retraced'] and mutated['shape'] == base['shape']  # stale program, old shape
    assert not rows['nodal_injections * 1.1 only']['retraced']
    assert rows['nodal_injections * 1.1 only']['max'] == pytest.approx(base['max'])
    # branch-outage entries scale by exactly 1.1; the generator-outage entries (an unscaled delta P) by less
    scaled = rows['nodal_injections and unsplit_flow * 1.1']['max']
    assert base['max'] < scaled <= 1.1 * base['max'] * (1 + 1e-9)
    assert not rows['max_mw_flow / 2, same sc']['retraced']
    assert rows['max_mw_flow / 2, same sc']['max'] == pytest.approx(base['max'])  # the old limits were baked in
    assert rows['max_mw_flow / 2, replace(sc)']['retraced']
    assert rows['max_mw_flow / 2, replace(sc)']['max'] == pytest.approx(2 * base['max'], rel=1e-6)
    # 4 and 8 topologies are both padded to batch_size_bsdf (16 here), so the core sees the same shapes
    batch = rows['batch of 8 topologies']
    assert batch['shape'][0] == 8 and batch['retraced'] == (si.solver_config.batch_size_bsdf < 8)


def test_batch_sweep_reports_throughput(si):
    rows = batch_sweep(si, n_topologies=16, bsdf_sizes=(4, 8), injection_sizes=(1,))
    assert len(rows) == 2 and all('error' not in r for r in rows), rows
    assert all(r['topologies_per_s'] > 0 and r['compile_s'] > 0 for r in rows)
