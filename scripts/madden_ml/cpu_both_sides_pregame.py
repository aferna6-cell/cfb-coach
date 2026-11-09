#!/usr/bin/env python3
"""Read-only BOTH-SIDES CPU-game preparation: real offense artifact + defense V2.

Does not modify either game DB, stage/install a book, activate a model, or claim
a full-playclock benchmark. Run on the player's laptop where the private
offensive structural artifact and current installed Madden books exist.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


def _counts(book: Mapping[str, Any] | None) -> dict[str, Any]:
    forms = (book or {}).get("formations") or {}
    return {"formations": len(forms), "plays": sum(len(x) for x in forms.values()),
            "formations_named": list(forms)}


def _fingerprint(path: Path) -> dict[str, Any]:
    b = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": sha256(b).hexdigest(), "bytes": len(b)}


def _books(db: Any) -> dict[str, Any]:
    from cfb_coach.madden.playbook import load_books
    return load_books(db)


def audit(*, db: Any, offense: Mapping[str, Any] | None = None,
          defense: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Evidence of stored locked books only, NOT an Xbox editor attestation."""
    installed = _books(db)
    sides = {}
    for side, proposal in (("offense", offense), ("defense", defense)):
        book = installed.get(side) or {}
        forms = book.get("formations") or {}
        locked = bool(book.get("locked_ts"))
        proposed = ((proposal or {}).get("book") or {}).get("formations") if side == "offense" else (
            (proposal or {}).get("formations"))
        matches = proposed is not None and forms == proposed
        sides[side] = {
            **_counts(book), "locked_in_cfb_coach": locked,
            "at_or_below_15_formations": 0 < len(forms) <= 15,
            "matches_proposal": matches,
            "model_proposal_supplied": proposed is not None,
            "ready_for_first_live_test": bool(locked and len(forms) <= 15 and matches),
            "physical_editor_confirmation_still_required": True,
        }
    return sides


def _safe_output(src: Path, outdir: Path) -> None:
    if outdir == src or src in outdir.parents:
        raise ValueError("output cannot reside inside the source database")
    if Path.home().joinpath(".cfb-coach") in outdir.parents:
        raise ValueError("output must be outside the active ~/.cfb-coach directory")


def _write_json(target: Path, value: dict[str, Any]) -> None:
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")


def generate(*, db: Any, franchise: str, opponent: str,
             offense_artifact: Path, outdir: Path,
             defense_artifact: Path | None = None,
             allow_cold_start: bool = False) -> dict[str, Any]:
    """Create both model-backed game-day proposals without falsely installing."""
    from cfb_coach.madden.model import defense_v2_pilot as v2
    from cfb_coach.madden.model import defense_v2_playbook, offense_designer
    from cfb_coach.madden import catalog

    if not franchise or not opponent:
        raise ValueError("both explicit testing franchise and CPU opponent required")
    if not offense_artifact.is_file():
        raise FileNotFoundError("actual learned offensive model artifact is required")
    src = Path(db.path).resolve()
    if src == Path.home().joinpath(".cfb-coach/madden27.db").resolve():
        raise ValueError("refusing default primary Madden database")
    outdir = outdir.resolve()
    _safe_output(src, outdir)
    if outdir == offense_artifact.resolve() or (defense_artifact and outdir == defense_artifact.resolve()):
        raise ValueError("do not write a plan over a model artifact")

    art = offense_designer.load_design_artifact(offense_artifact)
    # The offensive designer is the actual portfolio planner used by this
    # development stack, not a separate generic / artificially chosen book.
    offense = offense_designer.design_offense(
        db, opponent_id=opponent, max_formations=15, artifact=art,
        priors_on=True, skip_adjustment_design=True,
    )
    if not ((offense.get("book") or {}).get("formations")):
        raise ValueError("real offensive model did not return a playbook")

    verified, verified_audit = v2.verified_rows(db.conn, franchise=franchise)
    if defense_artifact:
        if not defense_artifact.is_file():
            raise FileNotFoundError(defense_artifact)
        defender = json.loads(defense_artifact.read_text(encoding="utf-8"))
        if defender.get("trained_franchise") != franchise or defender.get("schema") != v2.SCHEMA:
            raise ValueError("defense V2 artifact/franchise mismatch")
        defender_source = _fingerprint(defense_artifact)
    elif allow_cold_start and not verified:
        defender = v2.cold_start(franchise)
        defender_source = {"kind": "PRIOR_ONLY_NO_EXECUTED_DATA", "trained_snaps": 0}
    else:
        raise ValueError(
            "defense V2 artifact missing: supply --defense-artifact or "
            "--allow-cold-start for a new franchise with zero verified executions"
        )

    defense = defense_v2_playbook.propose(
        catalogue={b: catalog.book_formations("defense", b)
                   for b in catalog.book_names("defense")},
        model=defender, max_formations=15,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    _write_json(outdir / "offense_proposal.json", offense)
    _write_json(outdir / "defense_proposal.json", defense)
    if defender.get("cold_start") and not defense_artifact:
        _write_json(outdir / "defense_cold_start_model.json", defender)
    installed_audit = audit(db=db, offense=offense, defense=defense)
    report = {
        "status": "PREGAME_PROPOSALS_NOT_INSTALLED",
        "franchise": franchise, "cpu_opponent": opponent,
        "db": str(src),
        "offensive_model": _fingerprint(offense_artifact),
        "defensive_model": defender_source,
        "defensive_prior_only": bool(defender.get("cold_start")),
        "defensive_verified_executions": len(verified),
        "defensive_evidence_audit": verified_audit,
        "offense_proposal": _counts(offense["book"]),
        "defense_proposal": _counts(defense),
        "stored_book_audit": installed_audit,
        "ready_to_kickoff": False,
        "reason": ("Plan is NOT installed; offensive end-to-end loaded model latency, "
                   "defensive end-to-end latency and physical Madden book checks "
                   "remain to be established."),
        "steps": [
            "Review both model-selected full formation lists and source books.",
            "Build those exact books in Madden and confirm the installation in CFB Coach.",
            "Re-run the read-only installation match audit.",
            "Benchmark the active offense (306-play model when applicable) and V2 defense at <=150ms.",
            "For a cold-start V2, label it prior-only; log every actually executed defensive snap and optional adjustment.",
            "Only then play CPU with both sides enabled; check each live model/fallback attribution."
        ],
    }
    _write_json(outdir / "cpu_both_sides_pregame_report.json", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--franchise", required=True)
    p.add_argument("--cpu-opponent", required=True)
    p.add_argument("--offense-artifact", type=Path,
                   help="required for generating a new offense model proposal")
    p.add_argument("--defense-artifact", type=Path)
    p.add_argument("--allow-cold-start", action="store_true",
                   help="Explicitly permit untrained defensive prior in a NEW test franchise")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--check-only", action="store_true",
                   help="read existing proposals and compare against locked installed books")
    args = p.parse_args()

    from cfb_coach.db import CoachDB
    db = CoachDB(args.db, read_only=True)
    try:
        if args.check_only:
            offense = json.loads((args.out_dir / "offense_proposal.json").read_text(encoding="utf-8"))
            defense = json.loads((args.out_dir / "defense_proposal.json").read_text(encoding="utf-8"))
            matches = audit(db=db, offense=offense, defense=defense)
            result = {"status": "LOCKED_INVENTORY_MATCH_AUDIT",
                      "franchise": args.franchise, "stored_books": matches,
                      "both_books_match_model_proposals": all(
                          row["ready_for_first_live_test"] for row in matches.values()
                      ),
                      "physical_installation_not_independently_verifiable": True,
                      "loaded_150ms_latency_not_yet_measured": True}
        else:
            if args.offense_artifact is None:
                p.error("--offense-artifact required to produce model-generated offense book")
            result = generate(
                db=db, franchise=args.franchise, opponent=args.cpu_opponent,
                offense_artifact=args.offense_artifact, defense_artifact=args.defense_artifact,
                allow_cold_start=args.allow_cold_start, outdir=args.out_dir,
            )
    finally:
        db.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
