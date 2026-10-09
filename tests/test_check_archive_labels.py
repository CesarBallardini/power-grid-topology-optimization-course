import io
import json
import urllib.error
from pathlib import Path

import pytest

import check_archive_labels as cal

DOC = Path('doc.md')


def only(text: str) -> cal.Citation:
    citations = cal.extract_citations(DOC, text)
    assert len(citations) == 1
    return citations[0]


class TestExtraction:
    def test_label_before_link(self):
        c = only('3rd ed. 1996 **BORROW** [`bwb_Y0-AAK-944`](https://archive.org/details/bwb_Y0-AAK-944).')
        assert (c.identifier, c.label, c.line, c.text_mismatch) == ('bwb_Y0-AAK-944', 'BORROW', 1, None)

    def test_nearest_label_wins(self):
        text = 'Recent editions **PRINT-DISABLED**; 1st ed. 1987 **BORROW** [`id1`](https://archive.org/details/id1)'
        assert only(text).label == 'BORROW'

    def test_user_upload_label(self):
        text = (
            '4th ed. **PRINT-DISABLED**. ES: *Algebra lineal* (cited at UTN),\n'
            '    **USER-UPLOAD (ES)** [`algebra-lineal`](https://archive.org/details/algebra-lineal).'
        )
        c = only(text)
        assert (c.label, c.line) == ('USER-UPLOAD (ES)', 2)

    def test_label_on_previous_continuation_line(self):
        text = (
            '  - Lay, *Linear Algebra*. 2012 printing **BORROW**\n    [`isbn_1`](https://archive.org/details/isbn_1).'
        )
        assert only(text).label == 'BORROW'

    def test_unlabeled_link(self):
        assert only('Lecture videos FREE on archive.org: [`MIT6`](https://archive.org/details/MIT6).').label is None

    def test_label_of_previous_link_does_not_leak(self):
        text = '**BORROW** [`a`](https://archive.org/details/a), also [`b`](https://archive.org/details/b)'
        citations = cal.extract_citations(DOC, text)
        assert [(c.identifier, c.label) for c in citations] == [('a', 'BORROW'), ('b', None)]

    def test_label_of_previous_list_item_does_not_leak(self):
        text = '- Book A **FREE** online.\n- Book B [`b`](https://archive.org/details/b)'
        assert only(text).label is None

    def test_label_of_previous_table_cell_does_not_leak(self):
        text = '| Strang | **PRINT-DISABLED** (4th ed.) | ES scan [`es`](https://archive.org/details/es) |'
        assert only(text).label is None

    def test_label_in_same_table_cell(self):
        text = '| Strang | **PRINT-DISABLED** | ES, **USER-UPLOAD (ES)** [`es`](https://archive.org/details/es) |'
        assert only(text).label == 'USER-UPLOAD (ES)'

    def test_label_across_blank_line_does_not_leak(self):
        assert only('**FREE**\n\n[`b`](https://archive.org/details/b)').label is None

    def test_text_identifier_mismatch(self):
        c = only('**FREE** [`shown`](https://archive.org/details/real)')
        assert (c.identifier, c.text_mismatch) == ('real', 'shown')

    def test_plain_text_and_bare_links(self):
        text = (
            '**FREE** [the scan](https://archive.org/details/x/page/n5) and **BORROW** <https://archive.org/details/y>'
        )
        citations = cal.extract_citations(DOC, text)
        assert [(c.identifier, c.label, c.text_mismatch) for c in citations] == [
            ('x', 'FREE', None),
            ('y', 'BORROW', None),
        ]

    def test_non_details_links_ignored(self):
        assert cal.extract_citations(DOC, '**FREE** <https://archive.org/search?query=x>') == []


class TestClassify:
    @pytest.mark.parametrize(
        ('meta', 'status'),
        [
            ({}, cal.MISSING),
            ({'is_dark': True, 'metadata': {'collection': ['opensource']}}, cal.DARK),
            (
                {'metadata': {'collection': ['inlibrary', 'printdisabled'], 'access-restricted-item': 'true'}},
                cal.BORROW,
            ),
            ({'metadata': {'collection': ['printdisabled'], 'access-restricted-item': 'true'}}, cal.PRINT_DISABLED),
            ({'metadata': {'collection': 'printdisabled'}}, cal.PRINT_DISABLED),
            ({'metadata': {'collection': ['texts'], 'access-restricted-item': 'true'}}, cal.RESTRICTED),
            ({'metadata': {'collection': ['opensource'], 'language': 'spa'}}, cal.FREE),
            ({'metadata': {'collection': 'opensource'}}, cal.FREE),
        ],
    )
    def test_status(self, meta, status):
        assert cal.classify(meta) == status

    @pytest.mark.parametrize(
        ('label', 'status', 'ok'),
        [
            ('USER-UPLOAD (ES)', cal.FREE, True),
            ('USER-UPLOAD (ES)', cal.RESTRICTED, False),
            ('USER-UPLOAD (ES)', cal.BORROW, False),
            ('FREE', cal.FREE, True),
            ('FREE', cal.PRINT_DISABLED, False),
            ('BORROW', cal.BORROW, True),
            ('BORROW', cal.DARK, False),
            ('PRINT-DISABLED', cal.PRINT_DISABLED, True),
        ],
    )
    def test_label_matches(self, label, status, ok):
        assert (status in cal.ACCEPTED[label]) is ok


def fake_response(meta: dict) -> io.BytesIO:
    return io.BytesIO(json.dumps(meta).encode())


class TestFetchStatus:
    def test_request_has_user_agent_and_timeout(self, mocker):
        urlopen = mocker.patch('urllib.request.urlopen', return_value=fake_response({'metadata': {}}))
        assert cal.fetch_status('a b', timeout=5) == cal.FREE
        request = urlopen.call_args.args[0]
        assert request.full_url == 'https://archive.org/metadata/a%20b'
        assert request.get_header('User-agent') == cal.USER_AGENT
        assert urlopen.call_args.kwargs['timeout'] == 5

    def test_http_404_is_missing(self, mocker):
        error = urllib.error.HTTPError('u', 404, 'Not Found', None, None)
        mocker.patch('urllib.request.urlopen', side_effect=error)
        assert cal.fetch_status('x') == cal.MISSING

    def test_network_error_retried_then_error(self, mocker):
        sleep = mocker.patch('time.sleep')
        urlopen = mocker.patch('urllib.request.urlopen', side_effect=urllib.error.URLError('down'))
        assert cal.fetch_status('x', attempts=3) == cal.ERROR
        assert urlopen.call_count == 3
        assert [c.args[0] for c in sleep.call_args_list] == [2, 4]

    def test_transient_failure_recovers(self, mocker):
        mocker.patch('time.sleep')
        busy = urllib.error.HTTPError('u', 503, 'Busy', None, None)
        responses = [TimeoutError('slow'), busy, fake_response({'metadata': {'collection': ['inlibrary']}})]
        mocker.patch('urllib.request.urlopen', side_effect=responses)
        assert cal.fetch_status('x') == cal.BORROW

    def test_client_error_not_retried(self, mocker):
        error = urllib.error.HTTPError('u', 403, 'Forbidden', None, None)
        urlopen = mocker.patch('urllib.request.urlopen', side_effect=error)
        assert cal.fetch_status('x') == cal.ERROR
        assert urlopen.call_count == 1


class TestMain:
    def write_book(self, tmp_path: Path) -> Path:
        docs = tmp_path / 'docs'
        (docs / 'part').mkdir(parents=True)
        (docs / 'part' / 'ch.md').write_text(
            '**BORROW** [`lend`](https://archive.org/details/lend)\n'
            '**USER-UPLOAD (ES)** [`upload`](https://archive.org/details/upload)\n'
            'See [`video`](https://archive.org/details/video)\n',
            encoding='utf-8',
        )
        return docs

    def test_all_match(self, tmp_path, mocker, capsys):
        statuses = {'lend': cal.BORROW, 'upload': cal.FREE, 'video': cal.FREE}
        mocker.patch.object(cal, 'fetch_status', side_effect=lambda i, _timeout: statuses[i])
        assert cal.main([str(self.write_book(tmp_path))]) == 0
        out = capsys.readouterr().out
        assert 'UNLABELED video (actual FREE)' in out
        assert 'checked 3 citations (3 unique identifiers) in 1 files: 0 problems, 1 unlabeled' in out

    def test_mismatch_fails(self, tmp_path, mocker, capsys):
        statuses = {'lend': cal.FREE, 'upload': cal.DARK, 'video': cal.ERROR}
        mocker.patch.object(cal, 'fetch_status', side_effect=lambda i, _timeout: statuses[i])
        assert cal.main([str(self.write_book(tmp_path))]) == 1
        out = capsys.readouterr().out
        assert 'ch.md:1: MISMATCH lend stated BORROW, actual FREE' in out
        assert 'ch.md:2: MISMATCH upload stated USER-UPLOAD (ES), actual DARK' in out
        assert 'ch.md:3: ERROR could not check video' in out


@pytest.mark.network
def test_live_archive_org():
    # Lay's Linear Algebra, a controlled-lending item cited in chapter 2, and an identifier that cannot exist.
    assert cal.fetch_status('isbn_9781256144380') == cal.BORROW
    assert cal.fetch_status('this-identifier-does-not-exist-zz-0000') == cal.MISSING
