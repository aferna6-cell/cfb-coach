#!/usr/bin/env python3
"""Train a defensive V2 postgame challenger; compare to a frozen saved champion.

Read-only game database. Outputs a NEW candidate and report to an empty
directory, never installs, overwrites or promotes any live artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from cfb_coach.madden.model import defense_v2_live as live
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import defense_v2_actions as actions
from cfb_coach.madden.model import defense_v2_postgame as comparison


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-db", type=Path, required=True)
    p.add_argument("--franchise", required=True)
    p.add_argument("--champion", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    src = args.test_db.expanduser().resolve()
    if not src.is_file():
        p.error("test franchise DB not found")
    live_data = (Path.home() / ".cfb-coach").resolve()
    if live_data == src or live_data in src.parents:
        p.error("postgame training cannot read the primary live-data directory")
    dest = args.out_dir.expanduser().resolve()
    if dest == src or dest == live_data or live_data in dest.parents:
        p.error("refusing to write candidate into the primary live-data directory")
    if dest.exists():
        p.error("output directory already exists; use a new name (never overwrite)")
    champion = dict(live.load_model(args.champion))
    if champion.get("franchise") != args.franchise:
        p.error("champion trained on another franchise")
    with sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True) as conn:
        rows, stats = v2.verified_rows(conn, franchise=args.franchise)
        if not rows:
            p.error("no verified executed defensive training rows")
        counts = actions.fit_verified_adjustment_effects(conn, franchise=args.franchise)
    last = max(r.started_ts for r in rows)
    if not last or last <= str(champion["trained_through_started_ts"]):
        p.error("no newer verified game after champion's cutoff")
    report = comparison.compare_champion(champion, rows)
    challenger = v2.fit(rows)
    challenger.update({
        "franchise": args.franchise,
        "trained_sessions": sorted({r.session for r in rows}),
        "trained_through_started_ts": last,
        "executed_adjustments": counts,
    })
    payload = json.dumps(challenger, indent=2, sort_keys=True)
    report["candidate_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    report["training_audit"] = stats
    report["franchise"] = args.franchise
    report["candidate_training_games"] = challenger["n_games"]
    report["candidate_training_snaps"] = challenger["n_verified"]
    report["next_step"] = (
        "Keep champion; review evidence and predictive gate, test prospectively, "
        "then explicitly choose whether to change a separate testing-franchise model path."
    )
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "candidate.json").write_text(payload, encoding="utf-8")
    (dest / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                           encoding="utf-8")
    print(json.dumps({
        "candidate": str(dest / "candidate.json"),
        "comparison": str(dest / "comparison.json"),
        "prediction_gate": report["predictive_promotion_gate"],
        "champion_unchanged": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
