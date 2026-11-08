#!/usr/bin/env python3
"""Compile allowed provisional AceMadden diagram geometry into local QB-read hints.

Offline only: runs outside the 150ms live model loop, reads the real database
without modification, and writes anonymous route-family/location/depth features.
NO raw images, art paths/URLs, receiver identities, or button assignments.

Example:
  python3 scripts/madden_ml/prepare_cpu2_reads.py \
    --db ~/.cfb-coach/madden27.db \
    --catalog-store ~/cfb_cpu_game1_package/catalog_inputs \
    --provisional-export ~/cfb_cpu_game1_package/catalog_inputs/provisional_catalog_offense.v1.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

# python3 scripts/madden_ml/<script>.py places scripts/madden_ml first on
# sys.path. Without this, Python can silently import an older site-packages
# cfb_coach rather than this checkout. Both the compiler and CPU benchmark
# must run the SAME code under review as the interactive coach.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))
import tempfile

from cfb_coach.db import CoachDB
from cfb_coach.madden import catalog
from cfb_coach.madden.model.offense_catalog_index import load_offense_universe
from cfb_coach.madden.model.offense_geometry_decode import resolve_scoring_structure
from cfb_coach.madden.model.offense_inventory import installed_inventory, pairs_in_inventory
from cfb_coach.madden.model.offense_live_read_cache import SCHEMA, pairs_digest, default_path

ALLOWED = frozenset(("partial", "high_confidence_extracted"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=Path.home() / ".cfb-coach" / "madden27.db")
    p.add_argument("--catalog-store", type=Path, required=True)
    p.add_argument("--provisional-export", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    dbfile = args.db.expanduser()
    if not dbfile.is_file():
        p.error(f"Database not found: {dbfile}")
    db = CoachDB.open_read_only(dbfile)
    try:
        inv = installed_inventory(db)
    finally:
        db.close()
    if not inv.get("installed") or not inv.get("play_count"):
        p.error("No confirmed installed offensive inventory in the database")

    store = args.catalog_store.expanduser()
    feed = args.provisional_export.expanduser()
    if not (store / "catalog.jsonl").is_file() or not feed.is_file():
        p.error("Complete catalog.jsonl and provisional export are both required")

    index = load_offense_universe(ace_store=store, provisional_export=feed)
    by_key = {}
    for entry in index.entries:
        key = (catalog.norm(entry.formation), catalog.norm(entry.play))
        by_key.setdefault(key, []).append(entry)

    selected = []
    reason = Counter()
    for form, play in pairs_in_inventory(inv["formations"]):
        key = (catalog.norm(form), catalog.norm(play))
        structures = []
        for entry in by_key.get(key, ()):
            if not entry.model_use_permitted:
                continue
            geom = index.geometry_for(entry.listing_id)
            if geom is None:
                continue
            s = resolve_scoring_structure(
                form, play, geometry=geom,
                feature_tier=entry.feature_tier,
                model_use_permitted=True,
            )
            decoded = tuple(
                (spec.family, spec.side, spec.band, tier)
                for _, spec, tier in s.routes if tier in ALLOWED
            )
            if decoded:
                # Same-name duplicate listings can disagree about the play art.
                # Do not silently invent a resolution of conflicting graphs.
                structures.append((entry.feature_tier, decoded))
        if not structures:
            reason["missing_or_unusable_provisional_geometry"] += 1
            continue
        best_tier = "high_confidence_extracted" if any(
            t == "high_confidence_extracted" for t, _ in structures
        ) else "partial"
        choices = {tuple(sorted(routes)) for tier, routes in structures if tier == best_tier}
        if len(choices) != 1:
            reason["conflicting_diagrams_same_play"] += 1
            continue
        route_rows = next(iter(choices))
        selected.append({
            "formation": form,
            "play": play,
            "routes": [
                {"family": family, "side": side, "band": band, "tier": tier}
                for family, side, band, tier in route_rows
            ],
        })
    data = {
        "schema": SCHEMA,
        "pairs_digest": pairs_digest(pairs_in_inventory(inv["formations"])),
        "source_catalog_hash": index.catalog_hash,
        "source_geometry_hash": index.geometry_hash,
        "provenance": "anonymous provisional diagram geometry; not verified player routes",
        "plays": selected,
    }
    out = args.out.expanduser() if args.out else default_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Atomic private local output; never modify the source DB or commit art.
    fd, tmp = tempfile.mkstemp(prefix="offense_reads_", suffix=".json", dir=out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(json.dumps({
        "cache_path": str(out),
        "confirmed_installed_plays": inv["play_count"],
        "provisional_read_coverage": len(selected),
        "first_party_routes": "retained independently by the live read path",
        "unknown_or_ambiguous": inv["play_count"] - len(selected),
        "uncovered_reasons": dict(reason),
        "inventory_pairs_digest": data["pairs_digest"],
        "catalog_entries_indexed": index.n_indexed,
        "note": "No receiver identity or route timing verified; anonymous inferred reads only.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
