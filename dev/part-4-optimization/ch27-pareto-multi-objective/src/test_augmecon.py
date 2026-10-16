from augmecon import augmecon2, enumerate_front, lexmin


def test_payoff_table():
    assert lexmin(0, 1) == (8.0, 20.0)
    assert lexmin(1, 0) == (30.0, 9.0)


def test_augmecon2_finds_the_exact_front_with_few_milps():
    front, solved = augmecon2()
    assert front == enumerate_front()
    assert len(front) == 6
    assert solved < 21  # the bypass skips grid points
