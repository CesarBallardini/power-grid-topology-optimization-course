import brief


def test_switching_steps_of_the_node_breaker_topology():
    summary = brief.load(brief.NODE_BREAKER_RUN / 'topology_0' / 'orao_summary.json')
    steps = brief.switching_steps(summary)
    assert len(steps) == 3
    assert steps[0].startswith('1. Open breaker VL2_BREAKER')
    assert steps[1].startswith('2. Open disconnector L12_DISCONNECTOR_4_0')
    assert steps[2].startswith('3. Close disconnector L12_DISCONNECTOR_4_2')
    assert all('disconnector: no load current' in s for s in steps[1:])


def test_situation_and_proposals_read_the_metrics():
    sit = brief.situation(brief.load(brief.NODE_BREAKER_RUN / 'unsplit_ac_metrics.json'))
    assert sit['N-1 overload energy (MW)'] > 16 and sit['critical branches N-1'] == 2.0
    props = brief.proposals(brief.NODE_BREAKER_RUN)
    assert len(props) == 1 and props[0]['actions'] == [5] and props[0]['split substations'] == 1.0


def test_brief_text_has_the_four_sections():
    text = brief.brief()
    for heading in ('## Situation', '## Proposal topology_0', '### Switching steps', '## Assumptions and risks'):
        assert heading in text
    assert 'relative to the unsplit grid: 1.55' in text
