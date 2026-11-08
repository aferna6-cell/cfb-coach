"""Synthetic HTML, no network or third-party art."""
import json
from types import SimpleNamespace

import pytest
from scripts.madden_ml import ace_playbook_memberships as m


def test_parser_requires_exact_existing_formation_and_ignores_foreign_links():
    forms = {'offense|known': {'side': 'offense', 'formation_slug': 'known'},
             'offense|same': {'side': 'offense', 'formation_slug': 'same'},
             'defense|same': {'side': 'defense', 'formation_slug': 'same'}}
    html = ''.join(f'<a href="{url}">x</a>' for url in [
        '/playbooks/madden27/test/known', '/playbooks/madden27/test/same',
        '/playbooks/madden27/test/defense/same', '/playbooks/madden27/test/known-typo',
        'https://foreign.invalid/playbooks/madden27/test/known'])
    matched, unmatched, ambiguous = m.formation_links(html, 'madden27', 'test', forms)
    assert matched == ['defense|same', 'offense|known']
    assert unmatched == ['/playbooks/madden27/test/known-typo']
    assert ambiguous == ['/playbooks/madden27/test/same']


def test_duplicate_formation_reused_across_books_and_downloader_files_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'active_art_processes', lambda: [])
    forms = {'schema': 'private.ace_formations.v1', 'formations': {
        'offense|known': {'side': 'offense', 'formation_slug': 'known', 'plays': ['one'], 'playbooks': []}}}
    (tmp_path / 'formations.json').write_text(json.dumps(forms))
    (tmp_path / 'playbooks.json').write_text(json.dumps({'playbooks': {}}))
    for name in ('catalog.jsonl', 'authorization.json'):
        (tmp_path / name).write_text('sentinel')
    calls = []

    class Fetcher:
        def _get(self, path):
            calls.append(path)
            if path == '/playbooks/madden27':
                return b'<a href="/playbooks/madden27/a">A</a><a href="/playbooks/madden27/b">B</a>'
            return f'<a href="{path}/known">Known</a>'.encode()

    result = m.collect(SimpleNamespace(root=tmp_path), Fetcher())
    assert result['unique_formations'] == 1
    assert result['playbooks_visited'] == 2
    f = json.loads((tmp_path / 'formations.json').read_text())['formations']['offense|known']
    assert f['playbooks'] == ['madden27|a', 'madden27|b']
    books = json.loads((tmp_path / 'playbooks.json').read_text())['playbooks']
    assert books['madden27|a']['formations'] == ['offense|known']
    for name in ('catalog.jsonl', 'authorization.json'):
        assert (tmp_path / name).read_text() == 'sentinel'
    calls.clear()
    m.collect(SimpleNamespace(root=tmp_path), Fetcher())
    assert calls == ['/playbooks/madden27']


def test_live_art_guard_makes_zero_network_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'active_art_processes', lambda: [123])
    with pytest.raises(RuntimeError, match='alive'):
        m.collect(SimpleNamespace(root=tmp_path), None)


def test_unrecognized_layout_stops_without_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'active_art_processes', lambda: [])
    (tmp_path / 'formations.json').write_text('{"formations": {}}')
    (tmp_path / 'playbooks.json').write_text('{"playbooks": {}}')
    class Fetcher:
        def _get(self, path):
            return b'<html>layout unknown</html>'
    with pytest.raises(ValueError, match='no public book'):
        m.collect(SimpleNamespace(root=tmp_path), Fetcher())
    assert (tmp_path / 'playbooks.json').read_text() == '{"playbooks": {}}'


@pytest.mark.parametrize('code', [401, 403])
def test_restricted_robots_stops_without_any_book_requests(tmp_path, code):
    import urllib.error
    calls = []
    class Opener:
        def open(self, request, timeout=0):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(request.full_url, code, 'restricted', {}, None)
    with pytest.raises(PermissionError, match='robots.txt'):
        m.StrictFetcher(SimpleNamespace(root=tmp_path), opener=Opener(), delay=1.5)
    assert calls == ['https://acemadden.com/robots.txt']
