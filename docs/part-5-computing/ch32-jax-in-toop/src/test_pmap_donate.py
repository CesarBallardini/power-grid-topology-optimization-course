import ast

from pmap_donate import run_in_subprocess


def test_four_devices_and_donation_in_a_fresh_interpreter():
    report = ast.literal_eval(run_in_subprocess())
    assert len(report['devices']) == 4 and report['output sum'] == 2.0 * sum(range(12))
    # On CPU the donation is accepted but not honoured: the input stays readable (on GPU/TPU it is deleted).
    assert report['pmap input after donation'] == float(sum(range(12))) or report['pmap input is_deleted']
