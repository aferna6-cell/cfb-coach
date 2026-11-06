#!/usr/bin/env python3
"""Reproduce a provenance-safe research queue for the 93 custom Madden 27 offensive plays.

Uses a recorded custom-book snapshot and the existing read-only personal game backup.
This does NOT infer play-specific routes, release rules, quarterback keys or OL blocks.
The generated queue is a research manifest, never training labels.

Example:
python build_offense_completion_queue.py --snapshot installed_playbook_snapshot.json \
  --sources installed_source_catalog_reference.json --history-db backup.db --out ./output
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

PERSONNEL = {
    "Gun 5WR Tight": {"WR": 5, "TE": 0, "HB": 0, "QB": 1, "OL": 5, "source": "https://www.madden-school.com/playbooks/buccaneers/offense/"},
    "Gun Doubles Clamp Stack": {"WR": 3, "TE": 1, "HB": 1, "QB": 1, "OL": 5, "source": "https://madden.tools/playbooks/formation/shotgun/doubles-stack-clamp"},
    "Gun Trips X Nasty": {"WR": 3, "TE": 1, "HB": 1, "QB": 1, "OL": 5, "source": "https://madden.tools/playbooks/formation/shotgun/trips-x-nasty"},
    "Pistol Deuce Close": {"WR": 2, "TE": 2, "HB": 1, "QB": 1, "OL": 5, "source": "https://www.madden-school.com/playbooks/buccaneers/offense/"},
    "Pistol Trips": {"WR": 3, "TE": 1, "HB": 1, "QB": 1, "OL": 5, "source": "https://madden.tools/playbooks/formation/pistol/trips"},
}


def game_call_counts(history_db: Path | None) -> tuple[Counter, Counter]:
    if not history_db:
        return Counter(), Counter()
    db = sqlite3.connect(f"file:{history_db.resolve()}?mode=ro", uri=True)
    try:
        # Historically logged != verified executed: tracked separately.
        logs = Counter()
        verified = Counter()
        for formation, play, status, tier in db.execute(
            "SELECT formation, play, executed_status, executed_verification "
            "FROM snaps WHERE side='offense' AND formation IS NOT NULL AND play IS NOT NULL"
        ):
            key = (formation, play)
            logs[key] += 1
            if status == "identified" and tier == "verified":
                verified[key] += 1
        return logs, verified
    finally:
        db.close()


def build(snapshot: dict, sources: dict, logs: Counter, verified: Counter) -> list[dict]:
    rec = snapshot["sides"]["offense"]
    if rec.get("inventory_fingerprint") != "OFF-3C6D-E543":
        raise ValueError("wrong recorded offensive inventory fingerprint")
    if set(rec["formations"]) != set(PERSONNEL):
        raise ValueError("unexpected formations; do not rewrite the source manifest")
    result = []
    for formation, plays in rec["formations"].items():
        src_book = rec["formation_sources"][formation]
        f = sources["books"]["offense"][src_book]["formations"][formation]
        if not set(plays).issubset(set(f["plays"])):
            raise ValueError(f"source mismatch for {formation}")
        for play in plays:
            uid = hashlib.sha256(json.dumps(["Madden NFL 27", src_book, formation, play], ensure_ascii=False).encode()).hexdigest()[:20]
            per = PERSONNEL[formation]
            result.append({
                "play_id": uid, "side": "offense", "source_book": src_book,
                "formation": formation, "play": play, "formation_page": f["url"],
                "personnel": f"{per['WR']}WR/{per['TE']}TE/{per['HB']}HB",
                "personnel_evidence": per["source"],
                "expected_skill_slots": "|".join([f"WR{i}" for i in range(1, per["WR"] + 1)] +
                                               [f"TE{i}" for i in range(1, per["TE"] + 1)] +
                                               [f"HB{i}" for i in range(1, per["HB"] + 1)]),
                "qb_slots": 1, "ol_slots": 5,
                "recorded_snap_count": logs.get((formation, play), 0),
                "verified_execution_snap_count": verified.get((formation, play), 0),
                "play_art_capture": "MISSING", "play_art_sha256": "",
                "player_assignments_verified": 0,
                "exact_routes_status": "UNKNOWN", "blocking_status": "UNKNOWN",
                "ready_for_training": False,
            })
    result.sort(key=lambda x: (-x["verified_execution_snap_count"], -x["recorded_snap_count"], x["formation"], x["play"]))
    if len(result) != 93 or len({(x["formation"], x["play"]) for x in result}) != 93:
        raise ValueError("expected exactly 93 unique formation/play entries")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, default=Path(__file__).resolve().parents[2] / "cfb_coach/data/madden27/installed_playbook_snapshot.json")
    p.add_argument("--sources", type=Path, default=Path(__file__).resolve().parents[2] / "cfb_coach/data/madden27/installed_source_catalog_reference.json")
    p.add_argument("--history-db", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.history_db and a.out.resolve() == a.history_db.resolve().parent:
        raise ValueError("do not write research outputs alongside the live game database")
    a.out.mkdir(parents=True, exist_ok=True)
    logs, confirmed = game_call_counts(a.history_db)
    rows = build(json.loads(a.snapshot.read_text()), json.loads(a.sources.read_text()), logs, confirmed)
    report = {
        "schema": "madden27.offense_completion_audit.v1",
        "snapshot": "OFF-3C6D-E543", "total_plays": len(rows),
        "formations": {name: sum(x["formation"] == name for x in rows) for name in PERSONNEL},
        "verified_execution_snaps_matching_recorded_book": sum(x["verified_execution_snap_count"] for x in rows),
        "personnel_sets_identified": len(PERSONNEL),
        "formation_personnel_observed_from_external_reference": True,
        "all_11_assignments_verified": 0,
        "play_art_images_attached_to_local_game_database": 0,
        "route_or_blocking_assignments_claimed_from_play_names": 0,
        "external_route_data_training_enabled": False,
        "reason": "Exact 11-player route/blocking assignments require diagrams or first-party game evidence."
    }
    with (a.out / "OFFENSE_93_completion_queue.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.out / "OFFENSE_93_completion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()