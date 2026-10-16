import shm_fill


def test_a_small_segment_fills_and_the_child_exits_cleanly():
    assert shm_fill.fill_in_child(1) == 0


def test_describe_names_exit_codes_and_signals():
    assert shm_fill.describe(0) == 'exit 0'
    assert shm_fill.describe(1) == 'exit 1'
    assert shm_fill.describe(-2) == 'SIGINT'
    assert shm_fill.describe(-999) == 'signal 999'


def test_shm_bytes_is_none_or_positive():
    size = shm_fill.shm_bytes()
    assert size is None or size > 0
