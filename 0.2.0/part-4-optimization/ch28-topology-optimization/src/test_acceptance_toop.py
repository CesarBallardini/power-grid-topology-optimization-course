"""Compares the port with ToOp's ``evaluate_acceptance``; runs only where ToOp is importable (`make test-toop`)."""

import itertools

import pytest

pytestmark = pytest.mark.toop
pytest.importorskip('toop_engine_dc_solver')
from acceptance import evaluate_acceptance  # noqa: E402
from toop_engine_topology_optimizer.ac import scoring_functions  # noqa: E402
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics  # noqa: E402

VALUES = {
    'non_converging_loadflows': (0, 1, 2),
    'overload_energy_n_1': (0.0, 0.4, 10.0),
    'critical_branch_count_n_1': (0, 1, 3),
}


def _grid():
    keys = list(VALUES)
    for combo in itertools.product(*VALUES.values()):
        yield dict(zip(keys, combo, strict=True))


@pytest.mark.parametrize('critical_voltage', [False, True])
def test_port_matches_toop_on_a_grid(critical_voltage):
    cases = list(_grid())
    for split in cases:
        for unsplit in cases:
            s = {**split, 'disconnected_branches': 1, 'voltage_jump_count_n_1': 2, 'critical_va_diff_count_n_1': 1}
            u = {**unsplit, 'voltage_jump_count_n_1': 1, 'critical_va_diff_count_n_1': 1}
            theirs = scoring_functions.evaluate_acceptance(
                Metrics(fitness=0.0, extra_scores=s),
                Metrics(fitness=0.0, extra_scores=u),
                enable_critical_voltage_rejection=critical_voltage,
            )
            mine = evaluate_acceptance(s, u, critical_voltage=critical_voltage)
            if theirs is None:
                assert mine is None
            else:
                assert mine['criterion'] == theirs.criterion
                assert mine['value_after'] == pytest.approx(theirs.value_after)
                assert mine['value_before'] == pytest.approx(theirs.value_before)
                assert mine['threshold'] == pytest.approx(theirs.threshold)


def test_exercise_5_with_toop():
    split = Metrics(
        fitness=0.0,
        extra_scores={'non_converging_loadflows': 1, 'overload_energy_n_1': 0.4, 'critical_branch_count_n_1': 1},
    )
    unsplit = Metrics(
        fitness=0.0,
        extra_scores={'non_converging_loadflows': 2, 'overload_energy_n_1': 0.0, 'critical_branch_count_n_1': 0},
    )
    reason = scoring_functions.evaluate_acceptance(split, unsplit)
    assert reason.criterion == 'overload-energy' and reason.value_before == 0.0 and reason.threshold == 0.95
