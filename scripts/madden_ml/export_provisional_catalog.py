#!/usr/bin/env python3
"""Issue #132: write the PROVISIONAL catalog pack + audit from the private Ace store (offline, no network).

Writes only outside the store. Never touches catalog_semantics.export() or any live DB.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cfb_coach.madden.model import provisional_catalog as pc  # noqa: E402
from scripts.madden_ml.extract_downloaded_art import identity, installed_inventory, slug  # noqa: E402

# Recorded audits only; NOT aliases (no identity rewrite is applied).
# extract_downloaded_art.ALIASES may rewrite two Gun 5WR Tight names for art matching;
# those rewrites must never mark a listing installable or "exact".
ALIAS_AUDIT_NOTE = "installed name slug has no exact site slug; unresolved alias audit, no alias invented"
NEAR_MISS_NOTES = {
    "madden27|offense|gun-5wr-tight|l-spot": (
        "Gun 5WR Tight / L Spot (ASCII L, 0x4c) vs Ace listing z-spot / Z Spot: not an exact slug match. "
        "In-repo playbooks.json uses both L Spot and Z Spot as distinct play names in different books, "
        "including some of the same formations. No first-party Xbox evidence they are the same play; no alias invented."
    ),
    "madden27|offense|gun-5wr-tight|zl-spot-shake": (
        "Gun 5WR Tight / Zl Spot Shake (Z + lowercase L) vs Ace listing z-spot-shake / Z Spot Shake: "
        "not an exact slug match. No first-party Xbox evidence they are the same play; no alias invented."
    ),
    "madden27|offense|gun-5wr-tight|z-spot": (
        "Ace listing Z Spot / z-spot is a near-miss for installed snapshot Gun 5WR Tight / L Spot (ASCII L, 0x4c); "
        "not an exact match. L Spot and Z Spot appear as distinct strings in in-repo playbooks.json. No alias invented."
    ),
    "madden27|offense|gun-5wr-tight|z-spot-shake": (
        "Ace listing Z Spot Shake / z-spot-shake is a near-miss for installed snapshot Gun 5WR Tight / Zl Spot Shake; "
        "not an exact match. No alias invented."
    ),
}


def exact_installed_and_audits(inv, catalog_ids):
    """Use the snapshot's own play names. Never treat art-matching aliases as exact identities."""
    installed = set()
    audits = {}
    for r in inv:
        ident = (r["side"], slug(r["formation"]), slug(r["play"]))
        key = pc.play_key(*ident)
        if key in catalog_ids:
            installed.add(ident)
        else:
            audits[key] = NEAR_MISS_NOTES.get(key, f"{r['formation']} / {r['play']}: {ALIAS_AUDIT_NOTE}")
    for key, note in NEAR_MISS_NOTES.items():
        audits.setdefault(key, note)
    return installed, audits



def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--store', type=Path, default=Path('/home/box/cfb_acemadden'))
    a.add_argument('--out', type=Path, required=True)
    a.add_argument('--include-ace-derived-geometry', action='store_true')
    a.add_argument('--vlm-sidecar', type=Path)
    o = a.parse_args()
    if o.out.resolve().is_relative_to(o.store.resolve()):
        a.error('out must be outside the private store')
    cat = [json.loads(l) for l in (o.store / 'catalog.jsonl').open()]
    ids = {r['id'] for r in cat}
    inv = installed_inventory()
    installed, audits = exact_installed_and_audits(inv, ids)
    fb = {}
    for key, f in json.loads((o.store / 'formations.json').read_text())['formations'].items():
        fb[f"{f['side']}|{f['formation_slug']}"] = f.get('playbooks', [])
    vlm = json.loads(o.vlm_sidecar.read_text()) if o.vlm_sidecar else {}
    kw = dict(raw_dir=o.store / 'raw', extraction_dir=o.store / 'extractions', installed=installed,
              alias_audits=audits, formation_books=fb, include_ace_derived_geometry=o.include_ace_derived_geometry, vlm=vlm)
    o.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for side in ('offense', 'defense'):
        body = pc.export_provisional(cat, side=side, **kw)
        name = f'provisional_catalog_{side}.v1.jsonl'
        with (o.out / name).open('w') as f:
            f.write(json.dumps({k: v for k, v in body.items() if k != 'rows'}, separators=(',', ':')) + '\n')
            for r in body['rows']:
                f.write(json.dumps(r, separators=(',', ':')) + '\n')
        rows = body['rows']
        report[side] = {
            'file': name, 'sha256': hashlib.sha256((o.out / name).read_bytes()).hexdigest(),
            'n_indexed': body['n_indexed'], 'rows': len(rows),
            'art_url_present': sum(r['art_url_present'] for r in rows),
            'downloaded': sum(r['downloaded'] for r in rows), 'hash_ok': sum(r['art_hash_ok'] is True for r in rows),
            'hash_mismatch': sum(r['art_hash_ok'] is False for r in rows),
            'extracted': sum(r['extracted'] for r in rows),
            'downloaded_not_extracted': sum(r['downloaded'] and not r['extracted'] for r in rows),
            'url_less': [r['play_key'] for r in rows if not r['art_url_present']],
            'source_book_mapped': sum(r['source_book_mapped'] for r in rows),
            'source_book_unknown': sum(not r['source_book_mapped'] for r in rows),
            'installable': sum(r['installable'] for r in rows),
            'installed_match': dict(Counter(r['installed_book_match'] for r in rows)),
            'evidence_tier': dict(Counter(r['evidence_tier'] for r in rows)),
            'verified_assignments': 0, 'vlm_pass': sum(r['vlm_pass'] for r in rows),
            'formations': len({r['formation_slug'] for r in rows}),
        }
    files = sorted(p.name for p in (o.store / 'raw').glob('*.png'))
    report['raw_png_total'] = len(files)
    report['raw_unparsed'] = [f for f in files if identity(Path(f)) is None]
    report['raw_not_in_catalog'] = [f for f in files if identity(Path(f)) and pc.play_key(*identity(Path(f))) not in ids]
    report['unresolved_alias_audits'] = audits
    report['catalog_listings_total'] = len(cat)
    report['catalog_unique_ids'] = len(ids)
    (o.out / 'alias_audits.json').write_text(json.dumps({
        'schema': 'madden27.catalog_alias_audit.v1', 'policy': 'no aliases invented; near-misses are NOT the same play',
        'unresolved_installed_name_mismatches': audits,
        'raw_files_not_in_index': report['raw_not_in_catalog'],
        'literal_double_hyphen_slugs': [r['id'] for r in cat if '--' in r['play_slug']],
        'notes': {'madden27|offense|gun-doubles-clamp-stack|portland-ywheel':
                  'installed name slugs to portland-ywheel; site index lists portland-y-wheel; a stale raw file named portland-ywheel.png exists outside the index; not aliased',
                  **NEAR_MISS_NOTES,
                  'double_hyphen': 'filename parser now accepts literal hyphen runs in play slugs; slug kept verbatim (curl--outs), not normalised',
                  'installable_policy': 'installable only for exact snapshot slugs; extract_downloaded_art.ALIASES are art-matching near-misses, not exact identities'}}, indent=1) + '\n')
    (o.out / 'audit_report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({s: {k: v for k, v in report[s].items() if k != 'url_less'} for s in ('offense', 'defense')}, indent=1))
    print(json.dumps({k: report[k] for k in ('raw_png_total', 'raw_unparsed', 'raw_not_in_catalog', 'unresolved_alias_audits')}, indent=1))


if __name__ == '__main__':
    main()
