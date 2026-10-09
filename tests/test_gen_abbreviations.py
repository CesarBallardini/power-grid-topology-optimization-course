import gen_abbreviations as ga

GLOSSARY = """# Appendix D

## Acronyms

| Acronym | Expansion | Meaning | Spanish | Ch. |
|---|---|---|---|---|
| PTDF | Power transfer distribution factor | Flow change per injection | *factor* | 17 |
| AC | Alternating current | Sinusoidal | *CA* | 9 |
| KCL / KVL | Kirchhoff's current law / voltage law | Sums are zero | *leyes* | 8 |
| LCC / VSC | Line-commutated converter / voltage-source converter | HVDC converters | *convertidores* | 11 |
| NP-hard | Nondeterministic polynomial-time hard | Hard | *NP-difícil* | 23 |

## Terms

| Term | Meaning | Spanish | Ch. |
|---|---|---|---|
| Bridge | Cut edge | *puente* | 3 |
"""


class TestParseAcronymRows:
    def test_reads_only_the_acronym_table(self):
        rows = ga.parse_acronym_rows(GLOSSARY)

        assert [acronym for acronym, _ in rows] == ['PTDF', 'AC', 'KCL / KVL', 'LCC / VSC', 'NP-hard']

    def test_missing_section_gives_no_rows(self):
        assert ga.parse_acronym_rows('# Glossary\n\nNothing here.\n') == []


class TestBuildEntries:
    def test_skips_listed_and_non_alphabetic_acronyms(self):
        entries = ga.build_entries(ga.parse_acronym_rows(GLOSSARY))

        assert 'AC' not in entries
        assert 'NP-hard' not in entries
        assert entries['PTDF'] == 'Power transfer distribution factor'

    def test_overrides_fix_split_expansions(self):
        entries = ga.build_entries(ga.parse_acronym_rows(GLOSSARY))

        assert entries['KVL'] == "Kirchhoff's voltage law"
        assert entries['VSC'] == 'Voltage-source converter'

    def test_uneven_split_reuses_the_whole_expansion(self):
        entries = ga.build_entries([('ABC / XYZ', 'One shared expansion')])

        assert entries['ABC'] == entries['XYZ'] == 'One shared expansion'


class TestRender:
    def test_sorted_markdown_abbreviations(self):
        assert ga.render({'XLA': 'Accelerated Linear Algebra', 'GPU': 'Graphics processing unit'}) == (
            '*[GPU]: Graphics processing unit\n*[XLA]: Accelerated Linear Algebra\n'
        )


class TestMain:
    def test_check_detects_stale_file(self, tmp_path, monkeypatch):
        glossary = tmp_path / 'd-glossary.md'
        glossary.write_text(GLOSSARY, encoding='utf-8')
        output = tmp_path / 'includes' / 'abbreviations.md'
        monkeypatch.setattr(ga, 'GLOSSARY', glossary)
        monkeypatch.setattr(ga, 'OUTPUT', output)
        monkeypatch.setattr(ga, 'REPO_ROOT', tmp_path)

        assert ga.main(['--check']) == 1
        assert ga.main([]) == 0
        assert ga.main(['--check']) == 0
        assert '*[PTDF]: Power transfer distribution factor' in output.read_text(encoding='utf-8')
