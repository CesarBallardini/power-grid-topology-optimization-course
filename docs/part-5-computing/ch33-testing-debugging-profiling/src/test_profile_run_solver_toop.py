import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import profile_run_solver_toop  # noqa: E402


@pytest.fixture(scope='module')
def halves():
    if not profile_run_solver_toop.CASE57.exists():
        pytest.skip(f'{profile_run_solver_toop.CASE57} not found')
    return profile_run_solver_toop.load()


def test_second_call_is_much_faster_than_the_first(halves):
    di, sc = halves
    first, second = profile_run_solver_toop.timed_calls(di, sc)
    assert first > 5 * second


def test_profiler_writes_a_trace(halves, tmp_path):
    di, sc = halves
    profile_run_solver_toop.jax.clear_caches()  # the first test left the compiled program in the cache
    profile_run_solver_toop.timed_calls(di, sc, n_calls=1, trace_dir=tmp_path)
    files = profile_run_solver_toop.trace_files(tmp_path)
    assert any(f.endswith('.xplane.pb') for f in files), files
    totals = profile_run_solver_toop.trace_totals(tmp_path)
    assert totals['XLA compilation (backend_compile)'][1] >= 1
    assert totals['tracing (trace_to_jaxpr_dynamic)'][0] > 0
