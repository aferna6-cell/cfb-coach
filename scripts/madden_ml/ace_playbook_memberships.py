#!/usr/bin/env python3
"""After art exits, collect public book-to-existing-formation pointers only.

No art downloads, catalog writes, slug aliases, copied plays, or assignment claims.
Uses the existing robots-aware, 1.5-second rate-limited public Fetcher.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfb_coach.madden.model import ace_catalog as ac  # noqa: E402
from scripts.madden_ml.extract_downloaded_art import atomic_json  # noqa: E402


def active_art_processes():
    found = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args = p.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if any(Path(a).name == 'ace_catalog.py' for a in args) and 'art' in args:
            found.append(int(p.parent.name))
    return found


class Links(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.paths = set()
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return
        href = dict(attrs).get('href', '')
        u = urlsplit(href)
        if u.scheme not in ('', 'https') or u.netloc not in ('', 'acemadden.com'):
            return
        if u.path.startswith('/'):
            self.paths.add(u.path.rstrip('/'))


def book_links(html, game):
    base = f'/playbooks/{game}/'
    return {p[len(base):]: p for p in Links(html).paths
            if p.startswith(base) and '/' not in p[len(base):]
            and p[len(base):] not in ('', 'offense', 'defense', 'plays')}


def formation_links(html, game, book, formations):
    by_slug = defaultdict(list)
    for key, row in formations.items():
        by_slug[row['formation_slug']].append(key)
    found, unmatched, ambiguous = set(), set(), set()
    prefix = f'/playbooks/{game}/{book}/'
    for path in Links(html).paths:
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix):].split('/')
        if len(parts) == 1:
            keys = by_slug.get(parts[0], [])
        elif len(parts) == 2 and parts[0] in ac.SIDES:
            keys = [k for k in by_slug.get(parts[1], []) if formations[k]['side'] == parts[0]]
        else:
            continue
        if len(keys) == 1:
            found.add(keys[0])
        elif keys:
            ambiguous.add(path)
        else:
            unmatched.add(path)
    return sorted(found), sorted(unmatched), sorted(ambiguous)


class StrictFetcher(ac.Fetcher):
    """Do not inherit the legacy constructor's permissive robots-error fallback."""
    def __init__(self, *args, **kwargs):
        self.robots_failure = None
        super().__init__(*args, **kwargs)
        if self.robots_failure is not None:
            raise PermissionError('robots.txt unreadable/restricted; membership stopped') from self.robots_failure

    def _get(self, path, *, check_robots=True, binary=False):
        try:
            return super()._get(path, check_robots=check_robots, binary=binary)
        except Exception as exc:
            if not check_robots:
                self.robots_failure = exc
            raise


def collect(store, fetcher, game='madden27'):
    if active_art_processes():
        raise RuntimeError('art process still alive; membership traffic refused')
    fp, bp = store.root / 'formations.json', store.root / 'playbooks.json'
    formation_doc = json.loads(fp.read_text())
    book_doc = json.loads(bp.read_text())
    formations = formation_doc['formations']
    books = book_doc['playbooks']
    book_doc['schema'] = 'private.ace_playbooks.v1'
    index = fetcher._get(f'/playbooks/{game}').decode('utf-8', 'replace')
    links = book_links(index, game)
    if not links:
        raise ValueError('book index yielded no public book links; nothing guessed')
    for book, url in sorted(links.items()):
        if active_art_processes():
            raise RuntimeError('art process appeared; membership stopped')
        key = f'{game}|{book}'
        if books.get(key, {}).get('status') == 'visited':
            continue
        try:
            html = fetcher._get(url).decode('utf-8', 'replace')
        except FileNotFoundError:
            books[key] = {'game': game, 'book_slug': book, 'source_url': ac.BASE + url,
                          'status': 'not_found', 'formations': []}
        else:
            matched, unmatched, ambiguous = formation_links(html, game, book, formations)
            if not matched:
                raise ValueError(f'{book}: no unambiguous existing formation links; layout not understood, stopped')
            books[key] = {'game': game, 'book_slug': book, 'source_url': ac.BASE + url,
                          'status': 'visited', 'formations': matched, 'retrieved_at': ac.now(),
                          'unmatched_links': unmatched, 'ambiguous_links': ambiguous,
                          'basis': 'public team-book anchor targets; membership only, not assignment verification'}
            for ident in matched:
                formations[ident]['playbooks'] = sorted(set(formations[ident].get('playbooks', [])) | {key})
        # Both maps are resumable; rerun reconstructs missing reverse pointers below.
        for bkey, b in books.items():
            for ident in b.get('formations', []):
                if ident in formations:
                    formations[ident]['playbooks'] = sorted(set(formations[ident].get('playbooks', [])) | {bkey})
        formation_doc['note'] = 'unique formation records; public book pointers only; no assignment verification'
        book_doc['note'] = 'book entries reference unique existing formation keys; no copied formation/play bodies'
        atomic_json(fp, formation_doc)
        atomic_json(bp, book_doc)
        print('membership', book, books[key]['status'], len(books[key]['formations']), flush=True)
    return {'books_discovered': len(links), 'playbooks_visited': sum(b.get('status') == 'visited' for b in books.values()),
            'playbook_entries': len(books), 'unique_formations': len(formations),
            'formations_with_playbooks': sum(bool(f.get('playbooks')) for f in formations.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--game', default='madden27')
    p.add_argument('--delay', type=float, default=1.5)
    a = p.parse_args()
    if a.delay < 1.5:
        p.error('delay must be at least 1.5 seconds')
    if active_art_processes():
        p.error('art process still running; no membership requests made')
    store = ac.Store(a.store)
    ac.require_scope(store, 'metadata_bulk')
    fetcher = StrictFetcher(store, delay=a.delay)
    print(json.dumps(collect(store, fetcher, a.game)), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
