#!/usr/bin/env python3
"""Create a defensive V2 pregame proposal from the existing catalog and architect.

Read-only against the test database. Does not install or arm formations.
Requires a previously trained, same-franchise V2 artifact with verified outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from cfb_coach.madden.model import defense_v2_live
from cfb_coach.madden.model.playbook_architect import design_side


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-db", type=Path, required=True)
    parser.add_argument("--franchise", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-formations", type=int, default=15)
    parser.add_argument("--current-book", type=Path,
                        help="Optional current applied-book JSON, for model-driven transitions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.max_formations <= 15:
        parser.error("max formations must be between 1 and 15")
    if not args.test_db.is_file():
        parser.error("test database must exist")
    live = (Path.home() / ".cfb-coach").resolve()
    src = args.test_db.resolve()
    if live == src or live in src.parents:
        parser.error("refusing to use the primary ~/.cfb-coach game data directory")
    with sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True) as conn:
        try:
            rows = conn.execute("SELECT COUNT(*) FROM game_sessions WHERE dynasty=?",
                                (args.franchise,)).fetchone()
        except sqlite3.DatabaseError as exc:
            parser.error(f"cannot verify test franchise: {exc}")
        if not rows or not rows[0]:
            parser.error("no recorded test-franchise game session; cannot establish isolation")
    try:
        model = dict(defense_v2_live.load_model(args.model))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(f"invalid V2 model: {exc}")
    if model.get("franchise") != args.franchise or int(model.get("n_verified") or 0) < 1:
        parser.error("V2 model must have at least one verified execution from this franchise")
    current = None
    if args.current_book:
        current = json.loads(args.current_book.read_text(encoding="utf-8"))
        if not isinstance(current, dict) or not isinstance(current.get("formations"), dict):
            parser.error("--current-book must contain a formations object")
    evidence = {
        "verified_executions": int(model["n_verified"]),
        "labeled_rows": int(model["n_verified"]),
    }
    design = design_side(
        "defense", opponent_id=args.opponent, db=None,
        applied=current, evidence=evidence, art=None,
        max_formations=args.max_formations,
        evidence_quality=("empirical" if model["n_games"] >= 8 else "prior_driven"),
        defense_v2_model=model, opponent_type=("cpu" if args.opponent.lower().startswith("cpu") else "human"),
        remove_only_when_full=False,
    )
    if not design["book"]["formations"]:
        parser.error("no eligible source-backed defensive formations were found")
    result = {
        "schema": "madden.defense.v2.pregame.proposal.v1",
        "franchise": args.franchise, "opponent": args.opponent,
        "model_schema": model["schema"],
        "verified_training_snaps": model["n_verified"],
        "verified_training_games": model["n_games"],
        "policy": "proposed_only_not_installed",
        "physical_editor_installation_required": True,
        "model_supported_matchup_effects": "observational estimates; no proven counterfactual",
        "installation_steps": [
            "Create or edit a SEPARATE defensive custom book in Madden NFL 27.",
            "Add each listed formation from the indicated source playbook.",
            "Verify every listed stock play is actually available in the editor.",
            "Save, install and independently confirm the exact active defensive book.",
            "Confirm its formation+play fingerprint before enabling test-only V2.",
            "Never substitute missing Madden plays or invisible assignments.",
        ],
        "architect": design,
    }
    out = args.out.resolve()
    if out == src or live == out or live in out.parents:
        parser.error("output cannot overwrite the DB or enter the live data directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": "PROPOSAL_ONLY", "file": str(out),
        "fingerprint": design["fingerprint"],
        "formations": design["limits"]["formations"],
        "stock_plays": design["limits"]["plays"],
        "not_installed": True,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
