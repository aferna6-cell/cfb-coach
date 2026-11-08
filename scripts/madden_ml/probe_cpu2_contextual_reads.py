#!/usr/bin/env python3
"""Read-only counterfactual QB-read audit on the owner's INSTALLED Madden book.

This does NOT simulate coverages, prove a receiver is open, or decide a live
play. It checks whether the SAME actual installed pass play yields sensible,
clearly labeled read recommendations under different PRE-snap beliefs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from cfb_coach.db import CoachDB
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_presnap_advice as advice
from cfb_coach.madden.model.offense_decision import Decision, NO_ADJUSTMENT
from cfb_coach.madden.model.offense_inventory import installed_inventory, pairs_in_inventory
from cfb_coach.madden.model.offense_live_read_cache import (
    pairs_digest, structure_for_live_read,
)

SCENARIOS = (
    ("Cover 1 likely", {"cover_1_unspecified": 0.85, "cover_2_unspecified": 0.15}, 1, 10),
    ("Cover 2 zone likely", {"cover_2_unspecified": 0.85, "cover_1_unspecified": 0.15}, 1, 10),
    ("Cover 3 spot likely", {"cover_3_spot": 0.8, "cover_4_quarters": 0.2}, 1, 10),
    ("Cover 3 on third-and-12", {"cover_3_spot": 0.8, "cover_4_quarters": 0.2}, 3, 12),
    ("Mostly unknown", {"unknown": 0.9, "cover_2_unspecified": 0.1}, 1, 10),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path,
                        default=Path.home() / ".cfb-coach" / "madden27.db")
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()
    db = CoachDB.open_read_only(args.db.expanduser())
    try:
        inv = installed_inventory(db)
    finally:
        db.close()
    if not inv["installed"] or not inv["play_count"]:
        parser.error("No confirmed offensive playbook in database")
    digest = pairs_digest(pairs_in_inventory(inv["formations"]))
    results = []
    totals = {"installed": inv["play_count"], "routes_present": 0,
              "cover1_vs_cover2_change": 0, "short_vs_long_change": 0}
    for formation, play in pairs_in_inventory(inv["formations"]):
        stock = sm.structure_any(formation, play)
        structure = structure_for_live_read(
            formation, play, stock, installed_pairs_digest=digest,
        )
        if not structure.n_known:
            continue
        totals["routes_present"] += 1
        dec = Decision(
            formation=formation, play=play,
            adjustment=NO_ADJUSTMENT, probability=0.5,
            stock_probability=0.5, interval=(0.4, 0.6),
            final_structure=structure,
        )
        cards = {
            name: advice.build_card(
                dec, mix, situation={"down": down, "distance": distance}
            )
            for name, mix, down, distance in SCENARIOS
        }
        cover_change = (
            cards["Cover 1 likely"]["PRIMARY READ"]
            != cards["Cover 2 zone likely"]["PRIMARY READ"]
        )
        situational_change = (
            cards["Cover 3 spot likely"]["PRIMARY READ"]
            != cards["Cover 3 on third-and-12"]["PRIMARY READ"]
        )
        totals["cover1_vs_cover2_change"] += int(cover_change)
        totals["short_vs_long_change"] += int(situational_change)
        if cover_change or situational_change:
            results.append({
                "formation": formation, "play": play,
                "published_baseline": cards["Cover 1 likely"]["PUBLISHED BASELINE"],
                "scenario_primary_reads": {
                    name: card["PRIMARY READ"] for name, card in cards.items()
                },
                "basis": cards["Cover 2 zone likely"]["READ BASIS"],
                "coverage_change": cover_change,
                "distance_change": situational_change,
            })
    print(json.dumps({
        "note": "Counterfactual football-prior read suggestions, NOT a learned per-receiver success model.",
        "installed_inventory": inv["inventory_id"],
        "totals": totals,
        "changed_read_examples": results[:max(0, args.examples)],
        "interpretation": ("Not every play should flip its read; only when the "
                           "known routes and pre-snap situation support it."),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
