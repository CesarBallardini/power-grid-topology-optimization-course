from curative_timing import Case, verdict


def test_curative_in_time_is_enough():
    assert verdict(Case(1180, 940, 10)) == 'curative action is sufficient'


def test_late_curative_needs_preventive():
    assert verdict(Case(1180, 940, 20)).startswith('preventive action needed: the curative action arrives')


def test_flow_above_tatl_needs_preventive():
    assert verdict(Case(1300, 940, 10)).startswith('preventive action needed: the flow exceeds the TATL')


def test_boundaries_and_other_branches():
    assert verdict(Case(1250, 940, 15)) == 'curative action is sufficient'  # at the TATL, at the deadline
    assert verdict(Case(1000, 940, 0)).startswith('nothing to do')
    assert verdict(Case(1180, 1020, 10)).startswith('preventive action needed: the curative action does not')
