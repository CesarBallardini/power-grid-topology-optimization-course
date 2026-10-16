"""Lab B of Chapter 28 on a fresh ToOp run with AC validation; needs ToOp's interpreter (the course's container).

Run it from this folder::

    python lab_b_toop.py OUT_ROOT [node_breaker] [complex] [--reuse]

For each grid it copies ``grid.xiidm`` of the ToOp checkout (``$TOOP_REPO``, default the ``ToOp`` folder next to
this repository) into ``OUT_ROOT/<grid>`` and makes the ``run_pipeline`` call of
``notebooks/example2_small_grid_toop.ipynb`` with its settings (30 s of DC search with ToOp's default seed 42, one
descriptor ``split_subs`` with 4 cells, AC validation of the 5 best topologies), leaving out the two preprocessing
keywords Pydantic drops and with ``switching_distance`` added to the observed metrics, so that ``dc_info`` carries
it for Lab B4. Then it prints the Lab B1 table (``acceptance.run_table``) next to ToOp's own ``evaluate_acceptance``
verdicts, renders the before/after single-line diagrams of every split station with the
operated switches highlighted (Lab B3, written to ``topology_N/lab_b/``), and stores per topology the kind and voltage
level of every operated switch in ``switches.json`` for the hand-off of Lab B4 (``handoff.py``). With ``--reuse`` the
latest run already in ``OUT_ROOT`` is analysed again instead of running the pipeline.
"""

import datetime
import json
import os
import re
import shutil
import sys
import time
import types
from pathlib import Path

import pypowsybl as pp
import toop_engine_topology_optimizer.dc.main as dc_main
from acceptance import run_table
from omegaconf import DictConfig
from toop_engine_grid_helpers.powsybl.single_line_diagram.get_single_line_diagram_custom import (
    get_single_line_diagram_custom,
)
from toop_engine_interfaces.messages.preprocess.preprocess_commands import PreprocessParameters
from toop_engine_topology_optimizer.ac.scoring_functions import evaluate_acceptance
from toop_engine_topology_optimizer.benchmark.benchmark_utils import (
    PipelineConfig,
    get_paths,
    prepare_importer_parameters,
    run_pipeline,
    suppress_jax_logs,
)
from toop_engine_topology_optimizer.interfaces.messages.results import Metrics

TOOP_REPO = Path(os.environ.get('TOOP_REPO', Path(__file__).resolve().parents[5] / 'ToOp'))
GRIDS = {
    'node_breaker': TOOP_REPO / 'data' / 'grid_node_breaker' / 'grid.xiidm',
    'complex': TOOP_REPO / 'data' / 'complex_grid' / 'grid.xiidm',
}
HIGHLIGHTED = re.compile(r'class="[^"]*\bhighlight\b')  # the class ToOp adds to every highlighted element


class _FileSafeDatetime:
    @staticmethod
    def now() -> str:
        return datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')  # noqa: DTZ005


if os.name == 'nt':
    dc_main.datetime = types.SimpleNamespace(datetime=_FileSafeDatetime)


def latest_run(out_root: Path, name: str) -> tuple[Path, Path]:
    """The last run folder of a grid already processed in ``out_root`` and its data folder."""
    data_folder = out_root / name / 'grid'
    runs = sorted((data_folder / 'optimizer_snapshot').glob('run_*'), key=lambda p: int(p.name.split('_')[1]))
    return runs[-1], data_folder


def highlighted(svg: str) -> int:
    """How many SVG elements carry ToOp's ``highlight`` class (the colour itself only appears in the CSS)."""
    return len(HIGHLIGHTED.findall(svg))


def run(out_root: Path, name: str) -> tuple[Path, Path]:
    """The notebook's pipeline on one grid; returns the run folder (``res.json``, ``topology_N/``) and data folder."""
    iteration = out_root / name
    iteration.mkdir(parents=True, exist_ok=True)
    shutil.copy(GRIDS[name], iteration / 'grid.xiidm')
    pipeline_cfg = PipelineConfig(root_path=out_root, iteration_name=name, file_name='grid.xiidm', grid_type='powsybl')
    iteration_path, file_path, data_folder, snapshot_dir = get_paths(pipeline_cfg)
    results = str(iteration_path / 'results' / '{task_name}')
    dc_cfg = DictConfig(
        {
            'task_name': 'test',
            'fixed_files': [str(data_folder / pipeline_cfg.static_info_relpath)],
            'double_precision': None,
            'tensorboard_dir': results,
            'stats_dir': results,
            'summary_frequency': None,
            'checkpoint_frequency': None,
            'stdout': None,
            'double_limits': None,
            'num_cuda_devices': 1,
            'omp_num_threads': 1,
            'xla_force_host_platform_device_count': None,
            'output_json': str(iteration_path / 'results' / 'output.json'),
            'lf_config': {'distributed': False},
            'ga_config': {
                'runtime_seconds': 30,
                'me_descriptors': [{'metric': 'split_subs', 'num_cells': 4}],
                'observed_metrics': ['overload_energy_n_1', 'split_subs', 'switching_distance'],
                'n_worst_contingencies': 2,
            },
        }
    )
    ac_cfg = DictConfig(
        {'n_processes': 1, 'k_best_topos': 5, 'critical_voltage_jump_percent': 5.0, 'critical_va_diff_degree': 20.0}
    )
    importer_parameters = prepare_importer_parameters(file_path, data_folder)
    importer_parameters.area_settings.cutoff_voltage = 10
    paths = run_pipeline(
        pipeline_cfg=pipeline_cfg,
        importer_parameters=importer_parameters,
        preprocessing_parameters=PreprocessParameters(action_set_clip=2**10),
        dc_optim_config=dc_cfg,
        ac_validation_cfg=ac_cfg,
    )
    run_dir = paths[0].parent if paths else sorted(snapshot_dir.glob('run_*'))[-1]
    return run_dir, data_folder


def toop_verdicts(run_dir: Path) -> dict[str, str | None]:
    """ToOp's own ``evaluate_acceptance`` on the stored files: ``None`` (accepted) or the rejecting criterion."""
    unsplit = Metrics.model_validate(json.loads((run_dir / 'unsplit_ac_metrics.json').read_text(encoding='utf-8')))
    out = {}
    for path in sorted(run_dir.glob('topology_*/ac_metrics.json'), key=lambda p: int(p.parent.name.split('_')[1])):
        split = Metrics.model_validate(json.loads(path.read_text(encoding='utf-8')))
        reason = evaluate_acceptance(metrics_split=split, metrics_unsplit=unsplit)
        out[path.parent.name] = None if reason is None else reason.criterion
    return out


def operated_switches(orao_summary: dict) -> list[tuple[str, bool]]:
    """``[(switch id, open?), ...]`` in the order of the ORAO summary."""
    actions = orao_summary['forced-actions']['preventive-actions-list']['actions']
    return [(a['switchId'], a['open']) for a in actions if a['type'] == 'SWITCH']


def before_after(grid_xiidm: Path, topology_dir: Path) -> dict:
    """Lab B3 for one topology: highlighted before/after diagrams per split voltage level, switch kinds and states."""
    orao = json.loads((topology_dir / 'orao_summary.json').read_text(encoding='utf-8'))
    ids = [s for s, _ in operated_switches(orao)]
    before = pp.network.load(str(grid_xiidm))
    after = pp.network.load(str(topology_dir / 'modified_network.xiidm'))
    pp.loadflow.run_ac(before)
    pp.loadflow.run_ac(after)
    sw_before, sw_after = before.get_switches(), after.get_switches()
    switches = {
        s: {
            'kind': sw_before.loc[s, 'kind'],
            'voltage_level_id': sw_before.loc[s, 'voltage_level_id'],
            'open_before': bool(sw_before.loc[s, 'open']),
            'open_after': bool(sw_after.loc[s, 'open']),
        }
        for s in ids
    }
    (topology_dir / 'switches.json').write_text(json.dumps(switches, indent=2), encoding='utf-8')
    out_dir = topology_dir / 'lab_b'
    out_dir.mkdir(exist_ok=True)
    levels = {}
    for level in sorted({v['voltage_level_id'] for v in switches.values()}):
        sizes = []
        for tag, net in (('before', before), ('after', after)):
            svg = get_single_line_diagram_custom(net, level, highlight_grid_model_ids=ids)._content
            (out_dir / f'{level}_{tag}.svg').write_text(svg, encoding='utf-8')
            sizes.append((len(svg), highlighted(svg)))
        levels[level] = sizes
    return {'switches': switches, 'diagrams': levels}


def main() -> None:
    """Run the pipeline on the grids named (both by default) and print what Lab B1 and B3 ask for."""
    suppress_jax_logs()
    out_root = Path(sys.argv[1])
    reuse = '--reuse' in sys.argv
    for name in [a for a in sys.argv[2:] if a != '--reuse'] or list(GRIDS):
        start = time.perf_counter()
        run_dir, data_folder = latest_run(out_root, name) if reuse else run(out_root, name)
        print(f'{name}: run folder {run_dir}, pipeline {time.perf_counter() - start:.0f} s', flush=True)
        res = json.loads((run_dir / 'res.json').read_text(encoding='utf-8'))
        print(f'  DC: {res["iteration"]} epochs, {len(res["best_topos"])} topologies in best_topos')
        table = run_table(run_dir)
        table['toop_rejected_by'] = table['topology'].map(toop_verdicts(run_dir))
        print(table.to_string(index=False))
        for topology_dir in sorted(run_dir.glob('topology_*')):
            if not (topology_dir / 'orao_summary.json').exists():
                continue
            result = before_after(data_folder / 'grid.xiidm', topology_dir)
            print(f'  {topology_dir.name}: diagrams (characters, highlighted elements) {result["diagrams"]}')
            for switch, info in result['switches'].items():
                print(f'    {switch}: {info}')


if __name__ == '__main__':
    main()
