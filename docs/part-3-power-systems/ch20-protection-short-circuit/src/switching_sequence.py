"""Exercise 4 of Chapter 20: a switching sequence in a double-busbar, single-breaker station, checked step by step.

The station is a graph of nodes (busbars BB1 and BB2, the line bay's node between its breaker and its busbar
disconnectors, the coupler bay's node, the line) joined by switching devices. A breaker may switch under load. A
disconnector may only operate when no current flows through it, that is when its two ends are already joined by
another closed path (the voltage across it is zero and the current commutates), or when one end is dead (not
connected to any source or load). ``run`` applies a sequence and raises on the first forbidden operation; it also
reports the moments at which the line is in parallel across both busbars.
"""

from __future__ import annotations  # ToOp's interpreter is older than 3.14: ``double_busbar`` returns its class

from dataclasses import dataclass, field

import networkx as nx


@dataclass
class Device:
    """A switching device between two nodes; ``breaker`` tells whether it may switch under load."""

    a: str
    b: str
    breaker: bool
    closed: bool


@dataclass
class Station:
    """A double-busbar station with one line bay and one coupler bay; ``sources`` are the live nodes."""

    devices: dict[str, Device] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)

    @classmethod
    def double_busbar(cls) -> Station:
        """Line bay on busbar 1, coupler open, both busbars fed by other bays (modelled as live nodes)."""
        station = cls(sources={'BB1', 'BB2', 'LINE'})
        station.devices = {
            'Q1': Device('BAY', 'BB1', breaker=False, closed=True),  # busbar disconnector to BB1
            'Q2': Device('BAY', 'BB2', breaker=False, closed=False),  # busbar disconnector to BB2
            'CB': Device('BAY', 'LINE_SIDE', breaker=True, closed=True),  # line bay breaker
            'Q9': Device('LINE_SIDE', 'LINE', breaker=False, closed=True),  # line disconnector
            'QC1': Device('COUPLER', 'BB1', breaker=False, closed=False),  # coupler disconnectors
            'QC2': Device('COUPLER', 'BB2', breaker=False, closed=False),
            'CBC': Device('COUPLER', 'COUPLER_B', breaker=True, closed=False),  # coupler breaker
            'QC3': Device('COUPLER_B', 'BB2', breaker=False, closed=False),  # ToOp-style coupler bay: CB between
        }
        # a coupler bay is BB1 - QC1 - COUPLER - CBC - COUPLER_B - QC3 - BB2; QC2 is not used by the sequence
        del station.devices['QC2']
        return station

    def graph(self, without: str | None = None) -> nx.Graph:
        """The nodes joined by closed devices, optionally ignoring one device."""
        g = nx.Graph()
        for name, d in self.devices.items():
            g.add_nodes_from([d.a, d.b])
            if d.closed and name != without:
                g.add_edge(d.a, d.b)
        return g

    def live(self, node: str, g: nx.Graph) -> bool:
        """A node is live when a closed path connects it to a source."""
        return any(nx.has_path(g, node, s) for s in self.sources if s in g)

    def can_operate(self, name: str) -> tuple[bool, str]:
        """Whether the device may change state now, and why not."""
        d = self.devices[name]
        if d.breaker:
            return True, 'breaker'
        g = self.graph(without=name)
        if nx.has_path(g, d.a, d.b):
            return True, 'parallel path, no voltage across the disconnector'
        if not (self.live(d.a, g) and self.live(d.b, g)):
            return True, 'one side dead, no current'
        return False, 'would make or break load current'

    def line_on_both_busbars(self) -> bool:
        """Is the line connected to BB1 and BB2 at the same time (through the bay or the coupler)?"""
        g = self.graph()
        return nx.has_path(g, 'LINE', 'BB1') and nx.has_path(g, 'LINE', 'BB2')

    def busbars_coupled(self) -> bool:
        """Are BB1 and BB2 electrically one?"""
        return nx.has_path(self.graph(), 'BB1', 'BB2')


# Move the line bay from BB1 to BB2, then split the station (coupler open).
SEQUENCE = [
    ('close', 'QC1', 'coupler disconnector to BB1'),
    ('close', 'QC3', 'coupler disconnector to BB2'),
    ('close', 'CBC', 'coupler breaker: BB1 and BB2 are now paralleled'),
    ('close', 'Q2', 'bay disconnector to BB2 (make before break: Q1 still closed)'),
    ('open', 'Q1', 'bay disconnector to BB1 (its current commutated through Q2 and the coupler)'),
    ('open', 'CBC', 'coupler breaker: the station is split'),
    ('open', 'QC1', 'coupler disconnectors, dead'),
    ('open', 'QC3', ''),
]


def run(station: Station, sequence=SEQUENCE) -> list[dict[str, object]]:
    """Apply the sequence; raise ValueError on a forbidden operation; return one record per step."""
    log = []
    for action, name, note in sequence:
        ok, why = station.can_operate(name)
        if not ok:
            raise ValueError(f'{action} {name}: {why}')
        station.devices[name].closed = action == 'close'
        log.append(
            {
                'step': f'{action} {name}',
                'device': 'breaker' if station.devices[name].breaker else 'disconnector',
                'allowed because': why,
                'line on both busbars': station.line_on_both_busbars(),
                'busbars coupled': station.busbars_coupled(),
                'note': note,
            }
        )
    return log


def main() -> None:
    """Print the sequence with the checks, then show that opening Q1 without the coupler is refused."""
    for row in run(Station.double_busbar()):
        print(f'{row["step"]:10s} {row["device"]:12s} parallel={str(row["line on both busbars"]):5s}', end=' ')
        print(f'coupled={str(row["busbars coupled"]):5s} {row["allowed because"]}; {row["note"]}')
    try:
        run(Station.double_busbar(), [('close', 'Q2', ''), ('open', 'Q1', '')])
    except ValueError as e:
        print('wrong order, coupler open:', e)
    try:
        run(Station.double_busbar(), [('open', 'Q1', '')])
    except ValueError as e:
        print('opening Q1 under load:', e)


if __name__ == '__main__':
    main()
