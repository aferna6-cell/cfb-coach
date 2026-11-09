#!/usr/bin/env python3
"""Read-only CPU Game 2 audit of model attribution, per-snap evidence and input coverage.

Run on the owner's LOCAL DB, without modifying gameplay, model meta or the
installed book. Use --out to persist a machine-readable audit report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import sqlite3


def _json(value):
    try:
        return json.loads(value) if value else {}
    except (TypeError, ValueError):
        return {}


def _columns(conn, table):
    return {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}


def _artifact_fingerprint(value: str | None) -> dict:
    if not value:
        return {"path": None, "exists": False}
    candidate = Path(value).expanduser()
    result = {"path": str(candidate), "exists": candidate.is_file()}
    if not candidate.is_file():
        return result
    digest = hashlib.sha256()
    with candidate.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    result["sha256"] = digest.hexdigest()
    try:
        record = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(record, dict):
            for field in ("model_version", "kind", "evidence_quality", "n_train",
                          "n_supervised", "n_discounted_priors", "residual_active"):
                if field in record:
                    result[field] = record[field]
    except (ValueError, UnicodeError, OSError):
        result["artifact_json"] = "unreadable"
    return result


def audit(path: Path, *, game_id: str | None = None) -> dict:
    db = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        metas = {r["key"]: r["value"] for r in db.execute("SELECT key, value FROM meta")}
        if game_id is None:
            row = db.execute(
                "SELECT session_id FROM game_sessions "
                "WHERE opponent_id='cpu' AND play_count>0 ORDER BY started_ts DESC LIMIT 1"
            ).fetchone()
            game_id = row["session_id"] if row else None
        snaps = list(db.execute(
            "SELECT * FROM snaps WHERE session_id=? AND side='offense' ORDER BY id",
            (game_id or "",),
        ))
        decisions = list(db.execute(
            "SELECT * FROM ml_decisions WHERE game_id=? ORDER BY id", (game_id or "",)
        ))
        sessions = list(db.execute(
            "SELECT * FROM game_sessions WHERE session_id=?", (game_id or "",)
        ))
        sources = Counter(str(r["policy_source"] or "unknown") for r in decisions)
        statuses = Counter(str(r["shadow_status"] or "unknown") for r in decisions)
        game_clock_n = sum(r["game_clock"] is not None for r in snaps)
        scored_n = sum(r["score_for"] is not None and r["score_against"] is not None for r in snaps)
        has_verified = sum(
            r["executed_verification"] == "verified"
            and r["executed_status"] == "identified"
            and bool(r["executed_formation"]) and bool(r["executed_play"])
            for r in snaps
        )
        named_coverages = sum(bool(r["coverage_seen"]) for r in snaps)
        action_recommendations = sum(bool(r["recommended_adjustment_id"]) for r in snaps)
        executed_actions = sum(bool(r["executed_adjustment_id"]) for r in snaps)
        action_equal = sum(
            bool(r["recommended_adjustment_id"])
            and r["recommended_adjustment_id"] == r["executed_adjustment_id"]
            for r in snaps
        )
        fallback_markers = []
        for row in decisions:
            detail = _json(row["decision_json"])
            fallback_markers.append({
                "policy_source": row["policy_source"],
                "effective_mode": detail.get("effective_mode"),
                "fallback_reason": detail.get("fallback_reason"),
                "fallback_detail": detail.get("ml_status_detail"),
            })
        book = _json(metas.get("active_playbook_json"))
        book_offense = (book.get("applied") or {}).get("offense") or {}
        formations = book_offense.get("formations") or {}
        # Some legacy applied books store play lists as an alternate shape.
        playcount = sum(len(v) for v in formations.values() if isinstance(v, list))
        artifact_paths = {
            k: metas.get(k) for k in (
                "ml_experimental_artifact", "ml_route_aware_artifact",
            )
        }
        fingerprints = {
            k: _artifact_fingerprint(v)
            for k, v in artifact_paths.items()
        }
        meta_artifact_files = {
            k: row["exists"] for k, row in fingerprints.items()
        }
        report = {
            "schema": "cfb.cpu2.database_audit.v1",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "database": str(path.resolve()),
            "game_id": game_id,
            "scope": "read_only_existing_records",
            "configured_mode": metas.get("ml_mode"),
            "configured_ranker": metas.get("ml_offense_ranker"),
            "artifact_paths": artifact_paths,
            "artifact_files_exist_on_this_machine": meta_artifact_files,
            "artifact_fingerprints": fingerprints,
            "football_priors": {
                "stored": metas.get("ml_football_priors"),
                "effective_in_this_process": (
                    os.environ.get("CFB_COACH_FOOTBALL_PRIORS")
                    if os.environ.get("CFB_COACH_FOOTBALL_PRIORS") is not None
                    else metas.get("ml_football_priors")
                ),
            },
            "runtime_model_override": os.environ.get("CFB_COACH_OFFENSE_RANKER"),
            "installed": {
                "formation_count": len(formations),
                "play_count": playcount,
                "has_active_book": bool(book_offense),
            },
            "game": {
                "offensive_snaps": len(snaps),
                "verified_executions": has_verified,
                "named_postplay_coverages": named_coverages,
                "quarter_recorded": sum(r["quarter"] is not None for r in snaps),
                "clock_recorded": game_clock_n,
                "score_both_sides_recorded": scored_n,
                "adjustments_recommended": action_recommendations,
                "executed_adjustment_id_recorded": executed_actions,
                "recommended_id_matches_executed_id": action_equal,
                "decision_count": len(decisions),
                "policy_source_counts": dict(sources),
                "decision_status_counts": dict(statuses),
                "session_score": sessions[0]["score"] if sessions else None,
                "session_result": sessions[0]["result_wl"] if sessions else None,
            },
            "findings": [],
            "limitations": [
                "A matching executed_adjustment_id is not independent proof of button execution.",
                "This audit does not claim an artifact is valid, learned, calibrated or promoted.",
                "An old database backup cannot prove which model the user's CURRENT process loads.",
            ],
        }
        if decisions and sources.get("model", 0) == 0:
            report["findings"].append(
                "NO-GO: historical game contains no successful model-selected decisions"
            )
        if snaps and game_clock_n == 0:
            report["findings"].append(
                "NO-GO for clock-aware evaluation: game_clock missing from every offensive snap"
            )
        if len(snaps) and scored_n < len(snaps):
            report["findings"].append(
                "Partial score capture; do not infer missing values"
            )
        if sessions and sessions[0]["score"] == "14-14" and sessions[0]["result_wl"] == "win":
            report["findings"].append(
                "Session metadata inconsistent: 14-14 score and win label"
            )
        if metas.get("ml_offense_ranker") != "route_aware":
            report["findings"].append(
                "Live ranker is not stored as route_aware"
            )
        if any(not found for found in meta_artifact_files.values()):
            report["findings"].append(
                "An artifact configured in database is absent on this machine"
            )
        return report
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--game-id", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.db.expanduser().is_file():
        parser.error(f"database not found: {args.db}")
    report = audit(args.db.expanduser(), game_id=args.game_id)
    raw = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out.expanduser().write_text(raw + "\n", encoding="utf-8")
    print(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
