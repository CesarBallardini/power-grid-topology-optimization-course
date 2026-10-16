"""Preventive or curative? The PATL/TATL timing rule of Chapter 21, Exercise 1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    """One post-contingency situation: flows in A, times in minutes."""

    flow_after_outage: float
    flow_after_action: float
    activation_minutes: float
    patl: float = 1000.0
    tatl: float = 1250.0
    tatl_minutes: float = 15.0


def verdict(case: Case) -> str:
    """Which perimeter has to act.

    A curative action is enough when the post-contingency flow stays at or below the TATL, the action is in place
    before the TATL duration runs out and the flow it leaves is at or below the PATL. Otherwise a preventive action
    (or a faster curative one) is needed before the outage.
    """
    if case.flow_after_outage <= case.patl:
        return 'nothing to do: the post-contingency flow is within the PATL'
    if case.flow_after_outage > case.tatl:
        return 'preventive action needed: the flow exceeds the TATL, no time to act after the outage'
    if case.activation_minutes > case.tatl_minutes:
        return 'preventive action needed: the curative action arrives after the TATL duration'
    if case.flow_after_action > case.patl:
        return 'preventive action needed: the curative action does not bring the flow below the PATL'
    return 'curative action is sufficient'


def main() -> None:
    """The three situations of the exercise."""
    for case in (Case(1180, 940, 10), Case(1180, 940, 20), Case(1300, 940, 10)):
        print(case, '->', verdict(case))


if __name__ == '__main__':
    main()
