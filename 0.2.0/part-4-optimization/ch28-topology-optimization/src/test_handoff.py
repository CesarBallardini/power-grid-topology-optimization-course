import json

import handoff
from acceptance import run_table


def test_node_breaker_instructions_follow_the_orao_summary():
    folder = handoff.RUNS / 'node_breaker' / 'topology_0'
    lines = handoff.instructions(handoff.load(folder / 'orao_summary.json'), handoff.load(folder / 'switches.json'))
    assert lines == [
        'Open breaker VL2_BREAKER in VL2 (can break load current)',
        'Open disconnector L12_DISCONNECTOR_4_0 in VL2 (operate only without current)',
        'Close disconnector L12_DISCONNECTOR_4_2 in VL2 (operate only without current)',
    ]


def test_operations_equal_the_ac_switching_distance_and_exceed_the_dc_one():
    for grid in ('node_breaker', 'complex'):
        table = handoff.counts(handoff.RUNS / grid)
        split = table[table['operations'] > 0]
        assert (split['operations'] == split['ac_switching_distance']).all()
        assert (split['operations'] > split['dc_switching_distance']).all()
        assert (split['breakers'] + split['disconnectors'] == split['operations']).all()
    complex_table = handoff.counts(handoff.RUNS / 'complex')
    assert complex_table['operations'].tolist() == [12, 7, 5, 0]
    assert complex_table['dc_switching_distance'].tolist() == [8.0, 4.0, 4.0, 0.0]


def test_lab_b1_every_validated_topology_is_rejected_on_overload_energy():
    for grid in ('node_breaker', 'complex'):
        table = run_table(handoff.RUNS / grid)
        assert (table['rejected_by'] == 'overload-energy').all()
        assert table.loc[table['dc_rank'] == 1, 'topology'].item() == 'topology_0'
    table = run_table(handoff.RUNS / 'complex')
    unsplit = json.loads((handoff.RUNS / 'complex' / 'unsplit_ac_metrics.json').read_text(encoding='utf-8'))
    # the empty topology differs from the unsplit grid only by its PST setpoints, which the AC stage does not apply
    empty = table[table['topology'] == 'topology_3'].iloc[0]
    assert empty['overload_energy_n_1'] == unsplit['extra_scores']['overload_energy_n_1']
    # the AC ranking is nearly the reverse of the DC one: the DC winner has the second-worst AC overload energy
    assert table.sort_values('overload_energy_n_1')['dc_rank'].tolist() == [4.0, 3.0, 1.0, 2.0]
