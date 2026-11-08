#!/usr/bin/env python3
"""Offline, parallel-safe proposals from already downloaded art; never edits the catalog.

Only extractions/ is written inside --store. --watch rescans as the independent
art downloader adds files. No network, training, positional slots or verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfb_coach.madden.model import play_art_extract as pe  # noqa: E402

SNAPSHOT = ROOT / 'cfb_coach/data/madden27/installed_playbook_snapshot.json'
# Already owner-approved correspondences, also in build_offense_art_features.py.
ALIASES = {('Gun 5WR Tight', 'L Spot'): 'Z Spot',
           ('Gun 5WR Tight', 'Zl Spot Shake'): 'Z Spot Shake'}
SUMMARY_VERSION = 'downloaded_art_responsibilities.v1'


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def identity(path):
    parts = path.stem.split('__')
    if (path.suffix == '.png' and len(parts) == 3 and parts[0] in ('offense', 'defense')
            and all(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', p) for p in parts[1:])):
        return tuple(parts)
    return None


def installed_inventory(snapshot=SNAPSHOT):
    rows = []
    for side, book in json.loads(snapshot.read_text())['sides'].items():
        for formation, plays in book['formations'].items():
            for play in plays:
                site_play = ALIASES.get((formation, play), play) if side == 'offense' else play
                rows.append({'side': side, 'formation': formation, 'play': play,
                             'identity': (side, slug(formation), slug(site_play)),
                             'identity_basis': 'owner_approved_site_alias' if site_play != play else 'exact_slug'})
    return rows


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as out:
            json.dump(data, out, separators=(',', ':'))
            out.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def paths(out, ident):
    name = 'madden27__' + '__'.join(ident) + '.json'
    return out / name, out / 'responsibilities' / name, out / 'errors' / name


def proposed_responsibilities(ex, ident, sha):
    """Only carry extractor matches; defense has no per-marker zone association."""
    side, formation, play = ident
    flags = list(ex['flags'])
    if side == 'defense':
        flags += ['zone_marker_association_unknown', 'rush_colour_interpretation_unverified']
    markers = []
    for m in sorted(ex['players'], key=lambda x: x['index_left_to_right']):
        matches = []
        for i, r in enumerate(ex.get('routes', [])):
            if r['owner_marker_index'] == m['index_left_to_right']:
                matches.append({'kind': 'route_proposal', 'route_index': i,
                                'colour': r['colour'], 'suggested_shape': r['suggested_shape'],
                                'polyline': r['polyline'], 'owner_distance_px': r['owner_distance_px']})
        markers.append({'marker_index_left_to_right': m['index_left_to_right'],
                        'marker_shape': m['shape'], 'position_px': [m['x'], m['y']],
                        'slot': 'unknown', 'matched': matches or 'unknown',
                        'flags': flags + ([] if matches else ['marker_assignment_unknown'])})
    return {'schema': SUMMARY_VERSION, 'side': side, 'formation_slug': formation,
            'play_slug': play, 'play_key': 'madden27|' + '|'.join(ident),
            'art_sha256': sha, 'extractor_version': ex['extractor'], 'tier': pe.TIER,
            'needs_review': True, 'model_feature_eligible': False,
            'colour_meaning': 'UNVERIFIED (measured hue buckets only)',
            'markers': markers,
            'unmatched_routes': [r for r in ex.get('routes', []) if r['owner_marker_index'] is None],
            'zones_unassociated': ex.get('zones', []),
            'rush_arrows_unassociated': [{**r, 'colour': 'red', 'interpretation': 'unverified_extractor_proposal'}
                                        for r in ex.get('rush_arrows', [])],
            'defender_links_unassociated': ex.get('defender_links', []), 'flags': flags}


def process(path, out, installed, overlays=False):
    ident = identity(path)
    if ident is None:
        return 'unmatched_filename'
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    dest, summary, error = paths(out, ident)
    old = load(dest)
    if old.get('art_sha256') == sha and old.get('extractor') == pe.EXTRACTOR_VERSION:
        ex = old
        status = 'skipped'
    else:
        previous_error = load(error)
        if previous_error.get('art_sha256') == sha and previous_error.get('extractor_version') == pe.EXTRACTOR_VERSION:
            return 'failed'
        try:
            ex = pe.analyze(data, ident[0])
        except Exception as exc:
            atomic_json(error, {'file': path.name, 'art_sha256': sha, 'error': str(exc), 'type': type(exc).__name__,
                                'extractor_version': pe.EXTRACTOR_VERSION})
            return 'failed'
        ex.update({'id': 'madden27|' + '|'.join(ident), 'art_sha256': sha, 'file': path.name})
        atomic_json(dest, ex)
        status = 'extracted'
    old_summary = load(summary)
    if (old_summary.get('art_sha256') != sha or old_summary.get('schema') != SUMMARY_VERSION
            or old_summary.get('extractor_version') != pe.EXTRACTOR_VERSION):
        atomic_json(summary, proposed_responsibilities(ex, ident, sha))
    if overlays and ident in installed:
        overlay = dest.with_name(dest.stem + '_overlay.png')
        if status == 'extracted' or not overlay.exists():
            pe.render_overlay(data, ex).save(overlay)
    return status


def ordered_files(raw, installed, installed_only=False):
    def priority(path):
        ident = identity(path)
        if ident in installed:
            return 0 if ident[0] == 'offense' else 1
        return 2 if ident and ident[0] == 'defense' else 3
    return sorted((p for p in raw.glob('*.png') if not installed_only or identity(p) in installed),
                  key=lambda p: (priority(p), p.name))


def make_report(store, inventory, prefix):
    if prefix.resolve().is_relative_to(store.resolve()):
        raise ValueError('report prefix must be outside the downloader store')
    out = store / 'extractions'
    downloaded = {identity(p): p for p in (store / 'raw').glob('*.png') if identity(p)}
    status = {}
    for ident, path in downloaded.items():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        dest, summary, error = paths(out, ident)
        ex, sm = load(dest), load(summary)
        status[ident] = ('extracted' if ex.get('art_sha256') == sha and sm.get('art_sha256') == sha
                         else 'failed' if load(error).get('art_sha256') == sha else 'pending')
    full_indexed = Counter()
    for line in (store / 'catalog.jsonl').read_text().splitlines() if (store / 'catalog.jsonl').exists() else []:
        r = json.loads(line)
        full_indexed[r['side']] += 1
    counts = {'full_catalog': {}, 'installed': {}}
    for side in ('offense', 'defense'):
        for scope, ids, indexed in (
            ('full_catalog', [i for i in downloaded if i[0] == side], full_indexed[side]),
            ('installed', [r['identity'] for r in inventory if r['side'] == side],
             sum(r['side'] == side for r in inventory))):
            c = Counter(status.get(i, 'missing_image') for i in ids)
            counts[scope][side] = {'indexed': indexed, 'downloaded_with_art': sum(i in downloaded for i in ids),
                                   'extracted': c['extracted'], 'failed': c['failed'], 'pending': c['pending']}
    report = {'as_of': datetime.now().astimezone().isoformat(), 'counts': counts,
              'tier': pe.TIER, 'verified': 0, 'model_feature_eligible': 0,
              'missing_installed': [r for r in inventory if r['identity'] not in downloaded]}
    qa_path = out / 'qa' / 'sampler_qa.json'
    report['sampler_qa'] = load(qa_path)
    atomic_json(prefix.with_suffix('.json'), report)
    lines = ['# Private responsibility proposal progress', '', f"As of {report['as_of']} (America/New_York / ET).", '',
             '**Nothing is verified, geometry_reviewed, assignment_reviewed, or model_feature_eligible.**',
             'Measured line colours have unverified meaning. Slots, yards, reads and blocking remain unknown.',
             'Defense zones/rush candidates have no marker association; red-to-rush interpretation is unverified.',
             'Installed matching uses exact slugs plus the two previously owner-approved Gun 5WR Tight site aliases.', '',
             '| Scope | Side | Indexed | Downloaded | Extracted | Failed | Pending |',
             '|---|---|---:|---:|---:|---:|---:|']
    for scope, sides in counts.items():
        for side, c in sides.items():
            lines.append(f"| {scope} | {side} | {c['indexed']} | {c['downloaded_with_art']} | {c['extracted']} | {c['failed']} | {c['pending']} |")
    lines += ['', '## Installed offense proposed lines', '',
              'Marker indices are zero-based left-to-right detections, NOT positional slots; false/missed markers are possible.',
              '| Formation | Play | Marker / shape (slot unknown) | Suggested shape | Measured colour | Flags |',
              '|---|---|---|---|---|---|']
    for row in inventory:
        if row['side'] != 'offense':
            continue
        ident = row['identity']
        base = f"| {row['formation']} | {row['play']}"
        if status.get(ident) != 'extracted':
            lines.append(base + f" | unknown | {status.get(ident, 'image missing (no exact identity)')} | unknown | unknown |")
            continue
        summary = load(paths(out, ident)[1])
        for m in summary['markers']:
            matches = m['matched'] if isinstance(m['matched'], list) else [{}]
            for r in matches:
                lines.append(base + f" | {m['marker_index_left_to_right']} / {m['marker_shape']} | {r.get('suggested_shape', 'unknown')} | {r.get('colour', 'unknown')} | {', '.join(m['flags']) or 'none'} |")
        for r in summary['unmatched_routes']:
            lines.append(base + f" | unknown / unmatched | {r['suggested_shape']} | {r['colour']} | route_without_start_marker |")
    lines += ['', '## Missing installed images / exact identities', '']
    lines += [f"- {r['side']}: {r['formation']} / {r['play']} (no exact downloaded filename)." for r in report['missing_installed']]
    lines += ['', '## Visual sampler QA', '', json.dumps(report['sampler_qa'], indent=2), '',
              'Private raw art, overlays and proposals are not redistributed or committed. No training permission inferred.',
              'catalog.jsonl and authorization.json are read-only / untouched by this extractor.']
    prefix.with_name(prefix.name.replace('progress', 'report')).with_suffix('.md').write_text('\n'.join(lines) + '\n')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--installed-only', action='store_true')
    parser.add_argument('--overlays-installed', action='store_true')
    parser.add_argument('--report-prefix', type=Path)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least one second')
    out = args.store / 'extractions'
    out.mkdir(parents=True, exist_ok=True)
    # One incremental extractor per store, without touching downloader locks/state.
    with (out / '.extract_downloaded.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, 'another downloaded-art extractor owns this store\n')
        inventory = installed_inventory(args.snapshot)
        installed = {r['identity'] for r in inventory}
        while True:
            counts = Counter()
            for n, path in enumerate(ordered_files(args.store / 'raw', installed, args.installed_only), 1):
                counts[process(path, out, installed, args.overlays_installed)] += 1
                if n % 100 == 0:
                    print(datetime.now().astimezone().isoformat(), n, dict(counts), flush=True)
                    if args.report_prefix:
                        make_report(args.store, inventory, args.report_prefix)
            print('scan_complete', dict(counts), flush=True)
            if args.report_prefix:
                print('report', make_report(args.store, inventory, args.report_prefix), flush=True)
            if not args.watch:
                return 0
            time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
