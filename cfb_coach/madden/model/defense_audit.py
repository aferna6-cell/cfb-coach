"""Audit candidate defensive snaps before any play-specific training.

A logged defensive snap is a candidate, not a supervised example. Play-specific
training uses verified human executions with trustworthy outcomes only.
Opponent-level outcome risks may use labeled yards without claiming which
defensive play was run. Post-snap ``concept_seen`` is historical evidence, never
knowledge of the current offensive call.

This module does not invent snaps, interceptions, drive ids, or scores.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.defense_intelligence import (
    is_counted_defensive_snap,
    verified_human_rows,
)
from cfb_coach.outcome import parse_outcome

CANDIDATE_NOTE = (
    "Candidate defensive snaps are not automatically valid supervised examples. "
    "Uncertain executions are excluded from play-specific training."
)
REPORTED_CANDIDATES_UNVERIFIED = 178
ISSUE_75_LOGGED_INTS = 3
USER_REPORTED_INTS = 4


def _side(row: Mapping[str, Any]) -> str:
    text = str(row.get("side") or row.get("possession") or "").strip().lower()
    if text.startswith("d"):
        return "defense"
    if text.startswith("o"):
        return "offense"
    return text or "unknown"


def _game_key(row: Mapping[str, Any]) -> str:
    gid = str(row.get("game_id") or "").strip()
    return gid or "(missing game_id)"


def exclusion_reason(row: Mapping[str, Any]) -> str | None:
    """Why a defensive candidate cannot enter play-specific training, or None."""
    if _side(row) != "defense":
        return "not_defense"
    if str(row.get("opponent_type") or "").lower() != "human":
        return "not_human"
    if not str(row.get("opponent_id") or "").strip():
        return "missing_opponent_id"
    if not str(row.get("game_id") or "").strip():
        return "missing_game_id"
    if row.get("recommendation_only"):
        return "recommendation_only"
    if row.get("exclude") or row.get("invalid"):
        return "excluded_or_invalid"
    if str(row.get("executed_verification") or "") != "verified":
        return "execution_unverified"
    if str(row.get("executed_status") or "") != "identified":
        return "execution_not_identified"
    if not row.get("executed_play"):
        return "missing_executed_play"
    if row.get("supervised_eligible") is not True:
        return "not_supervised_eligible"
    if not row.get("label_available"):
        return "outcome_unlabeled"
    return None


def _turnover_kind(row: Mapping[str, Any]) -> str | None:
    raw = str(row.get("result") or row.get("outcome") or "")
    if not raw.strip():
        kind = str(row.get("outcome_kind") or row.get("kind") or "").lower()
        if kind in ("int", "interception"):
            return "int"
        if kind == "fumble":
            return "fumble"
        return None
    parsed = parse_outcome(raw)
    if parsed.kind in ("int", "interception"):
        return "int"
    if parsed.kind == "fumble":
        return "fumble"
    return None


def missing_fields(row: Mapping[str, Any]) -> list[str]:
    gaps: list[str] = []
    if not str(row.get("drive_id") or "").strip():
        gaps.append("drive_id")
    if not str(row.get("game_clock") or "").strip() and row.get("clock_seconds") in (None, ""):
        gaps.append("game_clock")
    if row.get("score_for") in (None, "") and row.get("score_us") in (None, ""):
        gaps.append("score")
    if str(row.get("executed_verification") or "") != "verified":
        gaps.append("executed_verification")
    if str(row.get("accepted") or "unknown") in ("", "unknown", "None"):
        gaps.append("accepted")
    return gaps


def audit_defensive_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    reported_candidate_count: int | None = REPORTED_CANDIDATES_UNVERIFIED,
    reported_ints: int | None = USER_REPORTED_INTS,
    issue_logged_ints: int | None = ISSUE_75_LOGGED_INTS,
) -> dict[str, Any]:
    """Supervised counts by game plus explanations for every exclusion."""
    defensive = [dict(r) for r in rows if _side(r) == "defense"]
    offensive = [dict(r) for r in rows if _side(r) == "offense"]
    candidates = [
        r for r in defensive
        if str(r.get("opponent_type") or "").lower() == "human" or not r.get("opponent_type")
    ]
    verified = [r for r in defensive if is_counted_defensive_snap(r)]
    verified_strict = [
        r for r in verified_human_rows(defensive) if _side(r) == "defense"
    ]

    by_game: dict[str, dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    for row in defensive:
        key = _game_key(row)
        bucket = by_game.setdefault(
            key,
            {
                "game_id": None if key == "(missing game_id)" else key,
                "opponent_id": str(row.get("opponent_id") or "") or None,
                "candidates": 0,
                "verified_supervised": 0,
                "excluded": 0,
                "exclusion_reasons": Counter(),
                "missing_fields": Counter(),
            },
        )
        bucket["candidates"] += 1
        reason = exclusion_reason(row)
        if reason is None:
            bucket["verified_supervised"] += 1
        else:
            bucket["excluded"] += 1
            bucket["exclusion_reasons"][reason] += 1
            excluded[reason] += 1
        for gap in missing_fields(row):
            bucket["missing_fields"][gap] += 1

    games = []
    for key, bucket in sorted(by_game.items()):
        games.append({
            **bucket,
            "exclusion_reasons": dict(bucket["exclusion_reasons"]),
            "missing_fields": dict(bucket["missing_fields"]),
        })

    ints_off = [r for r in offensive if _turnover_kind(r) == "int"]
    ints_def_forced = [r for r in defensive if _turnover_kind(r) == "int"]
    int_attribution = []
    for row in ints_off:
        int_attribution.append({
            "game_id": row.get("game_id") or None,
            "play": row.get("executed_play") or row.get("recommended_play") or row.get("play"),
            "result": row.get("result") or row.get("outcome"),
            "attribution": "unknown",
            "note": (
                "An interception is a recorded outcome. Individual reads or "
                "execution mistakes are not inferred from the playcall."
            ),
        })

    reported = reported_candidate_count
    in_memory = len(defensive)
    candidate_gap = None if reported is None else reported - in_memory

    logged_ints = len(ints_off)
    user_ints = reported_ints
    missing_int = None
    if user_ints is not None:
        missing_int = max(0, int(user_ints) - logged_ints)

    return {
        "schema": "madden.defense.audit.v1",
        "note": CANDIDATE_NOTE,
        "candidates": {
            "in_this_source": in_memory,
            "human_or_untyped": len(candidates),
            "reported_elsewhere": reported,
            "source_vs_reported_gap": candidate_gap,
            "reported_gap_explanation": (
                None if candidate_gap is None or candidate_gap == 0 else
                "The figure 178 is a candidate count from a franchise database "
                "that is not in this repository. It is not 178 supervised examples. "
                "Run this audit against the laptop Madden DB to obtain verified "
                "counts by game."
            ),
        },
        "supervised": {
            "verified_human_executions": len(verified),
            "verified_human_rows_helper": len(verified_strict),
            "play_specific_training": len(verified),
            "games_with_verified": sum(1 for g in games if g["verified_supervised"]),
        },
        "by_game": games,
        "exclusions": dict(excluded),
        "missing_data": {
            "drive_id": sum(g["missing_fields"].get("drive_id", 0) for g in games),
            "game_clock": sum(g["missing_fields"].get("game_clock", 0) for g in games),
            "score": sum(g["missing_fields"].get("score", 0) for g in games),
            "executed_verification": sum(
                g["missing_fields"].get("executed_verification", 0) for g in games
            ),
            "accepted": sum(g["missing_fields"].get("accepted", 0) for g in games),
        },
        "interceptions": {
            "logged_offensive_ints": logged_ints,
            "issue_75_logged_ints": issue_logged_ints,
            "user_reported_ints": user_ints,
            "unlogged_user_gap": missing_int,
            "defense_forced_ints": len(ints_def_forced),
            "attribution": int_attribution,
            "reconciliation": (
                "Issue #75 records three offensive interceptions from logged snaps. "
                "The user reports four. The last logged snap was 42-42; the session "
                "final was 45-42, so three points (and possibly a turnover) were "
                "never written as a snap. Eighteen of 59 offensive decisions had no "
                "snap. This audit does not invent a fourth interception row. "
                "Attribution of any INT to playcall versus execution remains unknown."
            ),
        },
        "post_snap_not_used_as_pre_snap": True,
        "john_overfit_guard": (
            "Opponent-specific tables stay separate from general defensive "
            "knowledge. John rows shrink toward the general prior."
        ),
    }


def attach_audit(model: dict[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(model)
    out["training_audit"] = {
        "verified_defensive_snaps": audit["supervised"]["play_specific_training"],
        "games": [
            {
                "game_id": g["game_id"],
                "opponent_id": g["opponent_id"],
                "verified_supervised": g["verified_supervised"],
                "candidates": g["candidates"],
            }
            for g in audit["by_game"]
        ],
        "exclusions": audit["exclusions"],
        "candidates_in_source": audit["candidates"]["in_this_source"],
        "reported_candidate_count": audit["candidates"]["reported_elsewhere"],
    }
    return out
