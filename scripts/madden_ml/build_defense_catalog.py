#!/usr/bin/env python3
"""Rebuild the Madden 27 defensive play knowledge catalog and its review matrix (deterministic; no network).

    python scripts/madden_ml/build_defense_catalog.py            # write cfb_coach/data/madden27/defense_play_catalog.json + docs matrix
    python scripts/madden_ml/build_defense_catalog.py --check    # exit 1 if the committed files differ from a fresh build

Reads only the recorded defensive book snapshot and the lexicon module. Never touches a game database or a trained model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv  # noqa: E402

from cfb_coach.madden.model import defense_catalog as dc  # noqa: E402
from cfb_coach.madden.model import defense_semantics as ds  # noqa: E402

NEEDS_PATH = dc.ROOT / "docs" / "madden_ml" / "DEFENSE_SCREENSHOT_NEEDS.csv"
CAPTURE = ("Xbox pre-snap play-art screenshot (all 11 defenders visible with their zone/man/rush paths) + the play name and formation on screen; "
           "one image per play. Hash recorded in the catalog; nothing is verified until then.")


def write_screenshot_needs(cat) -> None:
    """Which first-party screenshots would verify the most, in priority order. Priority: installed plays whose name the post-play form
    collapses with many others (or drops entirely) first, because those are where verified detail has the most to add."""
    ch = ds.channel_audit()
    stored_for = {n: lab for lab, names in ch["stored_label_to_possible_typed_names"].items() for n in names}
    size = {n: len(names) for names in ch["stored_label_to_possible_typed_names"].values() for n in names}
    rows = []
    for e in cat["entries"]:
        n = e["raw_name"]
        lost = n in ch["dropped_names"]
        collapse = size.get(n, 0)
        tier = (1 if e["status"] == "installed" and (lost or collapse >= 5) else 2 if e["status"] == "installed" else 3)
        rows.append({"priority_tier": tier, "id": e["id"], "formation": e["raw_formation"] or "", "play": n, "status": e["status"],
                     "verification_level_now": e["verification"]["level"], "unknown_items": len(e["unknowns"]),
                     "stored_label_today": "(nothing stored)" if lost else stored_for.get(n, ""),
                     "other_calls_stored_as_the_same_label": max(0, collapse - 1), "capture_needed": CAPTURE})
    rows.sort(key=lambda r: (r["priority_tier"], -r["other_calls_stored_as_the_same_label"], -r["unknown_items"], r["id"]))
    with open(NEEDS_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the committed catalog equals a fresh build")
    args = ap.parse_args(argv)
    cat = dc.build_catalog()
    errs = dc.validate_catalog(cat)
    if errs:
        print("catalog INVALID:\n  " + "\n  ".join(errs[:20]))
        return 1
    if args.check:
        committed = dc.load_catalog() if dc.CATALOG_PATH.exists() else {}
        if committed.get("digest") != cat["digest"]:
            print(f"committed catalog digest {committed.get('digest')} != fresh build {cat['digest']}: rerun without --check")
            return 1
        print(f"OK catalog digest {cat['digest']} matches a fresh build")
        return 0
    dc.write_catalog(cat)
    dc.write_matrix(cat)
    write_screenshot_needs(cat)
    s = cat["summary"]
    print(f"wrote {dc.CATALOG_PATH.name}: {s['entries']} entries ({s['installed']} installed + {s['external']} external), "
          f"{s['unique_names']} unique names, ledger_known_slots={s['ledger_known_slots']}, digest {cat['digest']}")
    print("verification levels:", s["verification_levels"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
