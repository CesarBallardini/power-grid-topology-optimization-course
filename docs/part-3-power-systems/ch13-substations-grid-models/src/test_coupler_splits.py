from coupler_splits import COUPLERS, two_node_configurations


def test_chain_of_three_couplers_has_three_splits():
    found = two_node_configurations(COUPLERS, 4)
    assert len(found) == 3  # open exactly one coupler of the chain
    for is_open, parts in found:
        assert sum(is_open) == 1
        assert sorted(len(p) for p in parts) in ([1, 3], [2, 2])


def test_ring_of_couplers_needs_two_open():
    ring = [(0, 1), (1, 2), (2, 3), (3, 0)]
    found = two_node_configurations(ring, 4)
    assert len(found) == 6  # choose 2 of 4 couplers to open
    assert all(sum(is_open) == 2 for is_open, _ in found)
