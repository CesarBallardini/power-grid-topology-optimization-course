"""Lab Part B of Chapter 19 with ToOp: an AC N-1 on the node-breaker example grid, metrics recomputed with pandas.

Run it with ToOp's interpreter (its container or virtual environment) from this folder::

    python ac_n1_toop.py ../../../../../ToOp/data/grid_node_breaker/grid.xiidm

Without an argument the grid is looked up in ``$TOOP_REPO`` or in the ``../ToOp`` sibling of this repository. The
N-1 definition is built by hand (one monitored element per line, transformer and bus; one contingency per branch
plus the base case), the AC analysis runs through ``get_ac_loadflow_results`` and the metrics are recomputed from
the result tables and compared with ``compute_metrics``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pypowsybl
from toop_engine_contingency_analysis.ac_loadflow_service import get_ac_loadflow_results
from toop_engine_contingency_analysis.ac_loadflow_service.compute_metrics import compute_metrics
from toop_engine_interfaces.nminus1_definition import Contingency, GridElement, MonitoredElement, Nminus1Definition

if TYPE_CHECKING:
    import pandas as pd

BASECASE = 'BASECASE'


def default_grid_path() -> Path:
    """ToOp's ``data/grid_node_breaker/grid.xiidm``."""
    repo = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[4].parent / 'ToOp'))
    return repo / 'data' / 'grid_node_breaker' / 'grid.xiidm'


def n1_definition(net: pypowsybl.network.Network) -> Nminus1Definition:
    """Step 1: monitor every line, transformer and bus; outage every branch one at a time, plus the base case."""
    branches = [(i, 'LINE') for i in net.get_lines().index] + [
        (i, 'TWO_WINDINGS_TRANSFORMER') for i in net.get_2_windings_transformers().index
    ]
    monitored = [MonitoredElement(id=i, type=t, kind='branch') for i, t in branches]
    monitored += [MonitoredElement(id=i, type='BUS', kind='bus') for i in net.get_buses().index]
    contingencies = [Contingency(id=BASECASE, elements=[])]
    contingencies += [Contingency(id=i, elements=[GridElement(id=i, type=t, kind='branch')]) for i, t in branches]
    return Nminus1Definition(monitored_elements=monitored, contingencies=contingencies)


def run(net: pypowsybl.network.Network, definition: Nminus1Definition):
    """Step 2: the AC N-1 through ToOp's load-flow service; returns the polars result bundle."""
    return get_ac_loadflow_results(net, definition, job_id='lab', timestep=0, n_processes=1)


def frames(results) -> dict[str, pd.DataFrame]:
    """The result tables as pandas DataFrames."""
    return {
        'branch': results.branch_results.collect().to_pandas(),
        'node': results.node_results.collect().to_pandas(),
        'va_diff': results.va_diff_results.collect().to_pandas(),
        'converged': results.converged.collect().to_pandas(),
    }


def pandas_metrics(branch: pd.DataFrame, va_diff: pd.DataFrame, base_case: str = BASECASE) -> dict[str, float]:
    """Step 3: ``overload_energy_n_1``, ``overload_current_n_1``, ``critical_branch_count_n_1``, ``max_va_diff_n_1``.

    The MW limit of a row is |p| / loading, so its overload is |p| - |p| / loading (the same with the current
    ``i``); per branch the worst contingency and side counts, then the overloads are summed.
    """
    n1 = branch[branch.contingency != base_case]
    out = {}
    for field, name in (('p', 'overload_energy_n_1'), ('i', 'overload_current_n_1')):
        over = n1[field].abs() - (n1[field] / n1.loading).abs()
        worst = over.groupby([n1.timestep, n1.element]).max()
        out[name] = float(worst[worst > 0].sum())
    critical = n1[n1.loading.fillna(-1) > 1.0][['timestep', 'element']].drop_duplicates()
    out['critical_branch_count_n_1'] = float(len(critical))
    out['max_va_diff_n_1'] = float(va_diff[va_diff.contingency != base_case].va_diff.abs().max())
    return out


def expected_rows(net: pypowsybl.network.Network, n_contingencies: int) -> int:
    """Step 4: the row count the formula promises, two sides per line and transformer, three per 3-winding one."""
    return (
        2 * len(net.get_lines())
        + 2 * len(net.get_2_windings_transformers())
        + 3 * len(net.get_3_windings_transformers())
    ) * n_contingencies


def main() -> None:
    """Steps 1 to 4 on the node-breaker example, printing every number the solution quotes."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_grid_path()
    net = pypowsybl.network.load(str(path))
    definition = n1_definition(net)
    print(
        f'{len(net.get_lines())} lines, {len(net.get_2_windings_transformers())} transformers, '
        f'{len(net.get_buses())} buses, {len(net.get_switches())} switches; '
        f'{len(definition.monitored_elements)} monitored elements, {len(definition.contingencies)} contingencies'
    )
    results = run(net, definition)
    f = frames(results)
    print({k: len(v) for k, v in f.items()}, 'convergence:', f['converged'].status.value_counts().to_dict())
    print('branch result rows', len(f['branch']), 'of', expected_rows(net, len(definition.contingencies)), 'expected')
    own = f['branch'][f['branch'].contingency == f['branch'].element]
    print('rows where the outaged branch reports in its own contingency:', len(own))
    mine = pandas_metrics(f['branch'], f['va_diff'])
    toop = compute_metrics(results, base_case_id=BASECASE)
    for name, value in mine.items():
        print(f'  {name:26s} pandas {value:12.6f}   compute_metrics {toop[name]:12.6f}')
    worst = f['branch'][f['branch'].contingency != BASECASE].sort_values('loading', ascending=False).head(3)
    print('three most loaded N-1 rows:')
    print(worst[['contingency', 'element', 'side', 'p', 'i', 'loading']].to_string(index=False))
    va = f['va_diff']
    print(
        'va_diff rows above 20 deg:',
        int((va.va_diff.abs() > 20).sum()),
        ' above 10 deg:',
        int((va.va_diff.abs() > 10).sum()),
    )
    print(
        va.loc[
            va.va_diff.abs().sort_values(ascending=False).index[:3], ['contingency', 'element', 'va_diff']
        ].to_string(index=False)
    )


if __name__ == '__main__':
    main()
