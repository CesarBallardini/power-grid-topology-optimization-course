import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')

import ray_store_toop  # noqa: E402


def test_a_small_object_round_trips_through_the_store():
    result = ray_store_toop.object_store(put_mb=8)
    assert result['intact']
    assert result['object_store_mb'] > 0
    # The warning appears exactly when /dev/shm exists and is smaller than the store Ray wants
    if result['shm_mb'] is None:
        assert result['fallback_warning'] is None
    else:
        assert (result['fallback_warning'] is not None) == (result['shm_mb'] < result['object_store_mb'])
