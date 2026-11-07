"""Issue #127 audit: per-side coverage funnel, the prioritized review queue (CSV), and the name-hint accuracy check.
Usage: python scripts/madden_ml/audit_catalog_semantics.py --queue docs/madden_ml/catalog_review_queue_installed.csv
       python scripts/madden_ml/audit_catalog_semantics.py --import-reviews filled_queue.csv   # human reviews only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cfb_coach.madden.model import catalog_semantics as cs  # noqa: E402

COLS = ["priority", "play_key", "formation", "play", "slot", "task", "proposed", "proposed_side", "proposed_tier", "art_ref", "evidence_to_use",
        "decision", "assignment", "reviewer", "reviewed_at", "evidence_ref", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path)
    ap.add_argument("--import-reviews", type=Path)
    a = ap.parse_args()
    if a.import_reviews:
        print(f"imported {cs.import_review_csv(a.import_reviews)} new review rows")
    o, d = cs.offense_records(), cs.defense_records()
    q = cs.review_queue(o + d)
    if a.queue:
        with open(a.queue, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(q)
    print(json.dumps({"funnel": {"offense": cs.funnel(o), "defense": cs.funnel(d)}, "queue_items": len(q),
                      "queue_by_priority": {p: sum(x["priority"] == p for x in q) for p in (1, 2, 3)},
                      "name_hint_accuracy": {k: v for k, v in cs.name_hint_accuracy().items() if k != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
