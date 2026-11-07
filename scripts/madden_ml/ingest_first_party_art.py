#!/usr/bin/env python3
"""Turn a folder of play-art images (your own Xbox captures, or art you hold a licence for) into PROPOSED geometry + a review queue.

  python scripts/madden_ml/ingest_first_party_art.py extract --images DIR --side offense [--store ~/cfb_acemadden]
  python scripts/madden_ml/ingest_first_party_art.py review  --store ~/cfb_acemadden        # writes review_queue.csv + overlay pictures
  python scripts/madden_ml/ingest_first_party_art.py approve --store ~/cfb_acemadden --file reviewed.csv

File names: ``<formation-slug>__<play-slug>.png`` (the same convention as the catalogue). Nothing is verified automatically: `extract` writes
``tier = diagram_extracted_unreviewed``. `approve` promotes only rows a human marked ``approved=yes`` with a ``source`` of the form
user_screenshot:<file> / user_recording:<file> / licensed_export:<ref>; everything else stays unverified. Images and overlays stay in the private store.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model import ace_catalog as ac  # noqa: E402
from cfb_coach.madden.model import play_art_extract as pe  # noqa: E402

SOURCE_OK = ("user_screenshot:", "user_recording:", "licensed_export:")


def cmd_extract(a) -> int:
    store = ac.Store(a.store)
    recs = store.read()
    out = store.root / "extractions"
    out.mkdir(exist_ok=True)
    n = skipped = 0
    for img in sorted(Path(a.images).expanduser().glob("*.png")):
        if "__" not in img.stem:
            skipped += 1
            continue
        f, p = img.stem.split("__", 1)
        rid = ac.record_id(a.game, a.side, ac.slug(f), ac.slug(p))
        data = img.read_bytes()
        ex = pe.analyze(data, a.side)
        ex.update({"id": rid, "art_sha256": hashlib.sha256(data).hexdigest(), "file": img.name})
        (out / f"{ac.slug(rid)}.json").write_text(json.dumps(ex, indent=1), encoding="utf-8")
        pe.render_overlay(data, ex).save(out / f"{ac.slug(rid)}_overlay.png")
        if rid in recs:
            recs[rid]["art_sha256"] = ex["art_sha256"]
            recs[rid]["content_hash"] = ac.content_hash(recs[rid])
        else:
            skipped += 1
        n += 1
    store.write_atomic(recs)
    print(f"extracted {n} images into {out} ({skipped} not matched to a catalogue record or not named formation__play). Tier: diagram_extracted_unreviewed.")
    return 0


def cmd_review(a) -> int:
    store = ac.Store(a.store)
    rows = []
    for f in sorted((store.root / "extractions").glob("*.json")):
        ex = json.loads(f.read_text())
        issues = list(ex["flags"])
        c = ex.get("counts", {})
        if ex["side"] == "offense" and c.get("routes", 0) != c.get("markers_with_route", 0):
            issues.append("route/marker count mismatch")
        if ex["side"] == "defense" and not ex.get("zones") and not ex.get("rush_arrows"):
            issues.append("nothing recognised (man coverage art? new colour scheme?)")
        rows.append({"id": ex["id"], "side": ex["side"], "counts": json.dumps(c), "issues": "; ".join(issues) or "none flagged",
                     "overlay": f.name.replace(".json", "_overlay.png"), "approved": "", "source": "", "reviewer_notes": ""})
    q = store.root / "review_queue.csv"
    with q.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} plays queued: {q}  (open the overlay next to the original art; set approved=yes + source only for what you can confirm)")
    return 0


def cmd_approve(a) -> int:
    store = ac.Store(a.store)
    recs = store.read()
    ok = bad = 0
    for row in csv.DictReader(Path(a.file).open(encoding="utf-8")):
        rid = row["id"]
        ex_path = store.root / "extractions" / f"{ac.slug(rid)}.json"
        if row.get("approved", "").strip().lower() != "yes":
            continue
        if rid not in recs or not ex_path.is_file() or not row.get("source", "").startswith(SOURCE_OK):
            bad += 1
            print(f"skipped {rid}: needs a catalogue record, an extraction and source in {SOURCE_OK}", file=sys.stderr)
            continue
        ex = json.loads(ex_path.read_text())
        r = recs[rid]
        key = "offense" if r["side"] == "offense" else "defense"
        r[key]["assignments" if key == "offense" else "responsibilities"] = {k: ex[k] for k in ("routes", "players", "counts") if k in ex} if key == "offense" else \
            {k: ex[k] for k in ("zones", "rush_arrows", "defender_links", "players", "counts") if k in ex}
        r["evidence"] = {"tier": "diagram_verified", "basis": f"{row['source']}; reviewed by owner; geometry from {pe.EXTRACTOR_VERSION}; slot names NOT assigned by the extractor"}
        r["art_sha256"] = ex["art_sha256"]
        r["content_hash"] = ac.content_hash(r)
        ok += 1
    store.write_atomic(recs)
    print(f"promoted {ok} plays to diagram_verified ({bad} skipped)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(Path.home() / "cfb_acemadden"))
    ap.add_argument("--game", default="madden27")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract"); p.add_argument("--images", required=True); p.add_argument("--side", choices=ac.SIDES, required=True); p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("review"); p.set_defaults(fn=cmd_review)
    p = sub.add_parser("approve"); p.add_argument("--file", required=True); p.set_defaults(fn=cmd_approve)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
