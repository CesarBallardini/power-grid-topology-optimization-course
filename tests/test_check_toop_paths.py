from pathlib import Path

import pytest

import check_toop_paths as ctp

DOC = Path('doc.md')


class TestNormalize:
    @pytest.mark.parametrize(
        ('token', 'expected'),
        [
            ('dc_solver/jax/bsdf.py', 'dc_solver/jax/bsdf.py'),
            ('dc_solver/example_grids.py:987', 'dc_solver/example_grids.py'),
            ('optimizer/ac/select_strategy.py:120-145', 'optimizer/ac/select_strategy.py'),
            ('optimizer/ac/select_strategy.py:select_strategy', 'optimizer/ac/select_strategy.py'),
            (
                'packages/dc_solver_pkg/src/x/multi_outages.py:89,118,247',
                'packages/dc_solver_pkg/src/x/multi_outages.py',
            ),
            ('docs/architecture/', 'docs/architecture/'),
            ('notebooks/openrao.ipynb.', 'notebooks/openrao.ipynb'),
            ('importer/network_graph/),', 'importer/network_graph/'),
        ],
    )
    def test_paths(self, token, expected):
        assert ctp.normalize(token) == expected

    @pytest.mark.parametrize(
        'token',
        [
            'docs/topology_optimizer/ac/*.md',
            'dc_solver/jax/{bsdf,lodf}.py',
            'docs/<page>.md',
            'dc_solver/jax/bsdf.py and more',
            'jax.lax.select',
            'toop_engine_dc_solver/jax/bsdf.py',
            'PTDF',
        ],
    )
    def test_not_paths(self, token):
        assert ctp.normalize(token) is None


@pytest.mark.parametrize(
    ('token', 'target'),
    [
        ('dc_solver/jax/bsdf.py', 'packages/dc_solver_pkg/src/toop_engine_dc_solver/jax/bsdf.py'),
        ('optimizer/ac/', 'packages/topology_optimizer_pkg/src/toop_engine_topology_optimizer/ac/'),
        ('interfaces/asset_topology.py', 'packages/interfaces_pkg/src/toop_engine_interfaces/asset_topology.py'),
        ('grid_helpers/powsybl/x.py', 'packages/grid_helpers_pkg/src/toop_engine_grid_helpers/powsybl/x.py'),
        (
            'contingency/pandapower/',
            'packages/contingency_analysis_pkg/src/toop_engine_contingency_analysis/pandapower/',
        ),
        ('importer/network_graph/', 'packages/importer_pkg/src/toop_engine_importer/network_graph/'),
        ('docs/index.md', 'docs/index.md'),
        ('notebooks/openrao.ipynb', 'notebooks/openrao.ipynb'),
        ('packages/dc_solver_pkg/README.md', 'packages/dc_solver_pkg/README.md'),
    ],
)
def test_expand(token, target):
    assert ctp.expand(token) == target


def test_extract_references_lines_and_skips():
    text = 'Intro `PTDF` and `dc_solver/jax/bsdf.py:12`.\n\nSee `docs/*.md` then `notebooks/a.ipynb`, done.\n'
    refs = ctp.extract_references(DOC, text)
    assert [(r.line, r.token, r.target) for r in refs] == [
        (1, 'dc_solver/jax/bsdf.py', 'packages/dc_solver_pkg/src/toop_engine_dc_solver/jax/bsdf.py'),
        (3, 'notebooks/a.ipynb', 'notebooks/a.ipynb'),
    ]


@pytest.fixture
def toop(tmp_path: Path) -> Path:
    root = tmp_path / 'ToOp'
    jax = root / 'packages/dc_solver_pkg/src/toop_engine_dc_solver/jax'
    jax.mkdir(parents=True)
    (jax / 'bsdf.py').write_text('', encoding='utf-8')
    (root / 'docs/architecture').mkdir(parents=True)
    return root


@pytest.fixture
def book(tmp_path: Path) -> Path:
    docs = tmp_path / 'docs'
    (docs / 'part').mkdir(parents=True)
    (docs / 'part' / 'ch.md').write_text(
        'Code: `dc_solver/jax/bsdf.py:10` and `docs/architecture/`.\nMissing: `dc_solver/jax/lodf.py:lodf`.\n',
        encoding='utf-8',
    )
    return docs


def test_main_reports_missing(toop, book, capsys):
    assert ctp.main([str(book), '--toop', str(toop)]) == 1
    out = capsys.readouterr().out
    assert (
        'ch.md:2: missing `dc_solver/jax/lodf.py` -> packages/dc_solver_pkg/src/toop_engine_dc_solver/jax/lodf.py'
        in out
    )
    assert 'checked 3 references (3 unique paths) in 1 files' in out
    assert out.rstrip().endswith('1 missing')


def test_main_all_present_via_env(toop, book, monkeypatch, capsys):
    (book / 'part' / 'ch.md').write_text('`dc_solver/jax/bsdf.py` `docs/architecture/`\n', encoding='utf-8')
    monkeypatch.setenv('TOOP_REPO', str(toop))
    assert ctp.main([str(book)]) == 0
    assert '0 missing' in capsys.readouterr().out


def test_main_missing_checkout(tmp_path, book, capsys):
    assert ctp.main([str(book), '--toop', str(tmp_path / 'nowhere')]) == 2
    assert 'ToOp checkout not found' in capsys.readouterr().err


def test_default_toop_root(monkeypatch):
    monkeypatch.delenv('TOOP_REPO', raising=False)
    assert ctp.default_toop_root() == ctp.REPO_ROOT.parent / 'ToOp'
    monkeypatch.setenv('TOOP_REPO', '/somewhere/ToOp')
    assert ctp.default_toop_root() == Path('/somewhere/ToOp')
