"""Before/after single-line diagrams of a split station with the operated switches highlighted (Ch. 35, Lab 4, Ex. 3).

``diagrams(grid_xiidm, modified_xiidm, orao_summary)`` finds the voltage level of the switches named in the ORAO
summary and renders both networks with ``get_single_line_diagram_custom(..., highlight_grid_model_ids=...)``.
Run with ToOp's interpreter: ``python sld_toop.py <run folder> <data folder> [output folder]`` writes the two SVG
files and prints what they contain.
"""

import json
import re
import sys
from pathlib import Path

import pypowsybl as pp
from toop_engine_grid_helpers.powsybl.single_line_diagram.get_single_line_diagram_custom import (
    get_single_line_diagram_custom,
)


def operated_switches(orao_summary):
    """``[(switch id, open?), ...]`` in the order of the summary."""
    actions = orao_summary['forced-actions']['preventive-actions-list']['actions']
    return [(a['switchId'], a['open']) for a in actions if a['type'] == 'SWITCH']


def stations_of(net, switch_ids):
    """The voltage levels that hold the switches, one per split station, in sorted order."""
    switches = net.get_switches()
    return sorted({switches.loc[s, 'voltage_level_id'] for s in switch_ids if s in switches.index})


def highlighted(svg):
    """How many SVG elements carry the ``highlight`` class that ToOp adds to the highlighted ids.

    The colour ``#0050fcff`` itself appears only in the diagram's CSS, once per voltage class, so counting it says
    nothing about which elements are highlighted.
    """
    return len(re.findall(r'class="[^"]*\bhighlight\b', svg))


def diagrams(grid_xiidm, modified_xiidm, orao_summary, style='bright_mode'):
    """``({level: (before svg, after svg)}, switch states before, switch states after)`` for the split stations."""
    ids = [s for s, _ in operated_switches(orao_summary)]
    before, after = pp.network.load(str(grid_xiidm)), pp.network.load(str(modified_xiidm))
    pp.loadflow.run_ac(before)
    pp.loadflow.run_ac(after)
    svgs = {}
    for level in stations_of(before, ids):
        svg_before = get_single_line_diagram_custom(before, level, custom_style=style, highlight_grid_model_ids=ids)
        svg_after = get_single_line_diagram_custom(after, level, custom_style=style, highlight_grid_model_ids=ids)
        svgs[level] = (svg_before._content, svg_after._content)
    states = lambda net: {s: bool(net.get_switches().loc[s, 'open']) for s in ids}  # noqa: E731
    return svgs, states(before), states(after)


def main():
    """Render the diagrams of every validated topology of a run and report sizes and switch states."""
    run_dir, data_folder = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else run_dir / 'brief_sld'
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(run_dir.glob('topology_*/orao_summary.json')):
        summary = json.loads(path.read_text(encoding='utf-8'))
        svgs, open_before, open_after = diagrams(
            data_folder / 'grid.xiidm', path.parent / 'modified_network.xiidm', summary
        )
        for level, (before, after) in svgs.items():
            (out_dir / f'{path.parent.name}_{level}_before.svg').write_text(before, encoding='utf-8')
            (out_dir / f'{path.parent.name}_{level}_after.svg').write_text(after, encoding='utf-8')
            print(
                f'{path.parent.name}: station {level}, {len(before)} and {len(after)} characters of SVG, '
                f'{highlighted(before)} and {highlighted(after)} highlighted elements'
            )
        for switch_id, _ in operated_switches(summary):
            print(f'  {switch_id}: open {open_before[switch_id]} -> {open_after[switch_id]}')
    print(f'written to {out_dir}')


if __name__ == '__main__':
    main()
