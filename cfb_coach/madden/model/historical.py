"""Safe audit and retrospective confirmation for historical Franchise games.

Never invents games. Never auto-elevates recommendations into verified executions.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model.schema import ExecutedStatus, Verification
from cfb_coach.opponents import is_cpu_opponent


def audit_database(db: Any) -> dict[str, Any]:
    """Read-only summary of historical snaps / games / label quality."""
    rows = dataset_mod.build_rows(db=db)
    games: dict[str, dict[str, Any]] = {}
    rec_only = 0
    verified = 0
    uncertain = 0
    missing_fields: dict[str, int] = {}
    coverage_timing: dict[str, int] = {}
    by_side = {"offense": 0, "defense": 0}
    cpu_games: set[str] = set()
    human_games: set[str] = set()

    for row in rows:
        gid = str(row.get("game_id") or row.get("session_id") or "unknown")
        g = games.setdefault(
            gid,
            {
                "game_id": gid,
                "opponent_id": row.get("opponent_id"),
                "opponent_type": row.get("opponent_type"),
                "n_snaps": 0,
                "verified_executions": 0,
                "recommendation_only": 0,
            },
        )
        g["n_snaps"] += 1
        oid = str(row.get("opponent_id") or "")
        if oid and is_cpu_opponent(oid):
            cpu_games.add(gid)
            g["opponent_type"] = "cpu"
        elif oid:
            human_games.add(gid)
            g.setdefault("opponent_type", "human")

        side = "defense" if str(row.get("side") or "").startswith("d") else "offense"
        by_side[side] += 1

        elig = row.get("eligibility") or dataset_mod.classify_eligibility(row)
        if elig == dataset_mod.ELIGIBILITY_VERIFIED_EXECUTION:
            verified += 1
            g["verified_executions"] += 1
        elif elig == dataset_mod.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC:
            uncertain += 1
            g["recommendation_only"] += 1
            rec_only += 1
        elif elig == dataset_mod.ELIGIBILITY_RECOMMENDATION_ONLY:
            rec_only += 1
            g["recommendation_only"] += 1

        for key in ("down", "distance", "yardline", "result", "recommended_play"):
            if row.get(key) in (None, ""):
                missing_fields[key] = missing_fields.get(key, 0) + 1

        # Coverage is post-snap unless coverage_source was live at decision time.
        src = str(row.get("coverage_source") or row.get("data_source") or "unknown")
        if row.get("coverage_seen") or row.get("coverage_hint"):
            timing = "post_snap_observed" if row.get("coverage_seen") else "hint_or_unknown"
            if "live" in src:
                timing = "pre_snap_live_hint"
            coverage_timing[timing] = coverage_timing.get(timing, 0) + 1

    quality = dataset_mod.quality_report(rows)
    return {
        "n_rows": len(rows),
        "n_games": len(games),
        "cpu_games": len(cpu_games),
        "human_games": len(human_games),
        "verified_executions": verified,
        "outcome_uncertain_or_recommendation": rec_only + uncertain,
        "by_side": by_side,
        "missing_fields": missing_fields,
        "coverage_observation_timing": coverage_timing,
        "eligibility": quality.get("eligibility"),
        "games": sorted(games.values(), key=lambda g: g["game_id"]),
        "supervised_training_rows": quality.get("supervised_training_rows", 0),
        "note": (
            "Unverified recommendations are not treated as verified executions. "
            "Use ml confirm-execution with explicit evidence to verify historical snaps."
        ),
    }


def confirm_execution(
    db: Any,
    *,
    ml_snap_id: str,
    executed_formation: str,
    executed_play: str,
    evidence: str,
    executed_macro: str | None = None,
) -> dict[str, Any]:
    """Retrospectively mark a snap's executed play from user evidence.

    Requires non-empty ``evidence`` (recording note, recollection, etc.).
    Does not invent formation/play names beyond what the user supplies.
    """
    if not ml_snap_id or not executed_formation or not executed_play:
        raise ValueError("ml_snap_id, executed_formation, and executed_play are required")
    if not (evidence or "").strip():
        raise ValueError("evidence is required — never auto-verify without user evidence")

    snap = db.conn.execute(
        "SELECT id, session_id, result, ml_snap_id FROM snaps WHERE ml_snap_id = ?",
        (ml_snap_id,),
    ).fetchone()
    if snap is None:
        # Allow confirming by numeric snaps.id when ml_snap_id was never assigned.
        if str(ml_snap_id).isdigit():
            snap = db.conn.execute(
                "SELECT id, session_id, result, ml_snap_id FROM snaps WHERE id = ?",
                (int(ml_snap_id),),
            ).fetchone()
        if snap is None:
            raise LookupError(f"no snap found for {ml_snap_id!r}")

    def _col(row: Any, name: str, idx: int) -> Any:
        if hasattr(row, "keys"):
            try:
                return row[name]
            except (IndexError, KeyError):
                return None
        return row[idx] if len(row) > idx else None

    row_id = int(_col(snap, "id", 0))
    sid = _col(snap, "session_id", 1)
    result = _col(snap, "result", 2)
    existing_ml = _col(snap, "ml_snap_id", 3)
    note = f"retrospective_confirm: {(evidence or '').strip()[:500]}"

    db.update_snap(
        row_id,
        executed_status=ExecutedStatus.IDENTIFIED.value,
        executed_formation=executed_formation,
        executed_play=executed_play,
        executed_macro=executed_macro,
        executed_verification=Verification.VERIFIED.value,
        notes=note,
    )
    if hasattr(db, "log_ml_outcome"):
        db.log_ml_outcome(
            snap_id=str(existing_ml or ml_snap_id),
            game_id=sid,
            executed_status=ExecutedStatus.IDENTIFIED.value,
            executed_formation=executed_formation,
            executed_play=executed_play,
            executed_verification=Verification.VERIFIED.value,
            outcome={"result": result, "evidence": evidence, "source": "retrospective"},
            replace=True,
        )
    return {
        "ok": True,
        "snap_row_id": row_id,
        "ml_snap_id": ml_snap_id,
        "executed_formation": executed_formation,
        "executed_play": executed_play,
        "evidence": evidence,
    }


def discounted_prior_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows with outcomes but uncertain execution — for descriptive / discounted priors only.

    Explicitly marked so they never enter SUPERVISED_ELIGIBLE training as ground truth.
    """
    out: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        elig = row.get("eligibility") or dataset_mod.classify_eligibility(row)
        if elig == dataset_mod.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC and row.get("label_available"):
            row["prior_discount"] = 0.25
            row["prior_role"] = "descriptive_uncertain_execution"
            out.append(row)
    return out
