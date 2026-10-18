"""Audit candidate defensive snaps before they become supervised examples.

The count of logged defensive snaps is a candidate figure, not a training
count. CPU rows, recommendations, unverified executions, and unlabeled
outcomes stay out of play-specific training.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import FAMILIES, call_family
from cfb_coach.madden.model.defense_intelligence import (
    is_counted_defensive_snap,
    verified_human_rows,
)
from cfb_coach.outcome import parse_outcome

AUDIT_SCHEMA = "madden.defense.training_audit.v1"

# Issue #75 recorded three offensive interceptions from logged snaps. The
# Wild Card user reported four. Neither figure is a reconstructed snap.
WILDCARD_JOHN_ISSUE_75 = {
    "issue": 75,
    "logged_offensive_int": 3,
    "user_reported_offensive_int": 4,
    "logged_snaps_offense": 41,
    "decisions_offense": 59,
    "unlogged_decisions": 18,
    "superseded_decisions": 17,
    "last_logged_score": "42-42",
    "session_final_score": "45-42",
    "defense_snaps_established": 45,
    "defense_int_forced_logged": 1,
}


def exclusion_reason(row: Mapping[str, Any]) -> str | None:
    """Why a candidate row is not a play-specific supervised example."""
    if not isinstance(row, Mapping):
        return "not_a_row"
    side = str(row.get("side") or "").lower()
    if side and side != "defense" and not side.startswith("d"):
        return "not_defense"
    if str(row.get("opponent_type") or "").lower() != "human":
        return "cpu_or_unknown_opponent"
    if not str(row.get("opponent_id") or "").strip():
        return "missing_opponent_id"
    if not str(row.get("game_id") or "").strip():
        return "missing_game_id"
    if row.get("recommendation_only"):
        return "recommendation_only"
    if row.get("supervised_eligible") is not True:
        return "not_supervised_eligible"
    if str(row.get("executed_verification") or "") != "verified":
        return "unverified_execution"
    if str(row.get("executed_status") or "") != "identified":
        return "unidentified_execution"
    play = row.get("executed_play") or row.get("action_play")
    if not play:
        return "missing_executed_play"
    if not row.get("label_available"):
        return "unlabeled_outcome"
    success = str(row.get("success") or "").lower()
    if success not in ("true", "false"):
        return "unknown_success_label"
    family = call_family(str(play))
    if family not in FAMILIES:
        return "unrecognized_defensive_family"
    return None


def _game_key(row: Mapping[str, Any]) -> str:
    return str(row.get("game_id") or row.get("session_id") or "unknown")


def _parse_kind(row: Mapping[str, Any]) -> str:
    payload = row.get("outcome_json")
    if isinstance(payload, Mapping) and payload.get("kind"):
        return str(payload.get("kind"))
    result = row.get("result")
    if result:
        return parse_outcome(str(result)).kind
    return "unknown"


def audit_defensive_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Split candidates from verified play-specific training examples."""
    candidates = [
        dict(row) for row in rows
        if str(row.get("side") or "").lower() in ("defense", "d")
        or str(row.get("side") or "").lower().startswith("d")
    ]
    reasons: Counter[str] = Counter()
    by_game: dict[str, dict[str, int]] = defaultdict(
        lambda: {"candidates": 0, "supervised": 0, "excluded": 0}
    )
    supervised: list[dict[str, Any]] = []
    missing: dict[str, int] = Counter()
    for row in candidates:
        game = _game_key(row)
        by_game[game]["candidates"] += 1
        why = exclusion_reason(row)
        if why:
            reasons[why] += 1
            by_game[game]["excluded"] += 1
            continue
        supervised.append(row)
        by_game[game]["supervised"] += 1
    counted = [row for row in candidates if is_counted_defensive_snap(row)]
    verified = [
        row for row in verified_human_rows(rows)
        if str(row.get("side") or "") == "defense"
    ]
    for field in ("down", "distance", "yardline", "quarter", "game_clock", "drive_id"):
        missing[field] = sum(1 for row in candidates if row.get(field) in (None, ""))
    kinds = Counter(_parse_kind(row) for row in supervised)
    john = sum(
        1 for row in supervised
        if str(row.get("opponent_id") or "").strip().lower() == "john"
    )
    return {
        "schema": AUDIT_SCHEMA,
        "candidate_defensive_snaps": len(candidates),
        "counted_defensive_snaps": len(counted),
        "verified_human_defensive_rows": len(verified),
        "supervised_play_specific": len(supervised),
        "excluded": dict(reasons),
        "by_game": dict(sorted(by_game.items())),
        "missing_fields_on_candidates": dict(missing),
        "supervised_outcome_kinds": dict(kinds),
        "john_supervised": john,
        "non_john_supervised": len(supervised) - john,
        "opponent_specific_held_out": True,
        "note": (
            "Candidate snaps are not automatically supervised examples. "
            "Uncertain executions are excluded from play-specific training. "
            "John's rows stay in the opponent-specific store, not the general prior."
        ),
        "reported_178_is_not_a_training_count": True,
    }


def reconcile_interceptions(
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    logged_int: int | None = None,
    reported_int: int | None = None,
) -> dict[str, Any]:
    """Compare logged INT events with a user report. Never invent a snap."""
    offense = [
        row for row in (rows or ())
        if str(row.get("side") or "").lower().startswith("o")
    ]
    counted = 0
    attributions: Counter[str] = Counter()
    for row in offense:
        kind = _parse_kind(row)
        if kind != "int":
            continue
        counted += 1
        payload = row.get("outcome_json") if isinstance(row.get("outcome_json"), Mapping) else {}
        attr = str((payload or {}).get("attribution") or "unknown")
        attributions[attr] += 1
    issue = WILDCARD_JOHN_ISSUE_75
    logged = issue["logged_offensive_int"] if logged_int is None else logged_int
    reported = issue["user_reported_offensive_int"] if reported_int is None else reported_int
    gap = int(reported) - int(logged)
    unlogged = int(issue["unlogged_decisions"])
    return {
        "issue": issue["issue"],
        "logged_offensive_int": logged,
        "user_reported_offensive_int": reported,
        "gap": gap,
        "rows_with_kind_int": counted,
        "unlogged_decisions": unlogged,
        "last_logged_score": issue["last_logged_score"],
        "session_final_score": issue["session_final_score"],
        "invented_snap": False,
        "attribution": "unknown",
        "attribution_counts": dict(attributions),
        "note": (
            "Issue #75 recorded three offensive interceptions among 41 logged "
            "snaps. The user reported four. Eighteen decisions had no snap "
            "(17 superseded). The last logged snap is 42-42 against a final "
            "of 45-42. A fourth INT cannot be reconstructed from committed "
            "evidence. Individual reads or execution mistakes stay unknown; "
            "an interception is not assumed to be a bad playcall."
        ),
        "cannot_silently_rewrite_a_snap": True,
    }


def go_nogo_from_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    n = int(audit.get("supervised_play_specific") or 0)
    games = len(audit.get("by_game") or {})
    if n < 1:
        verdict = "NO-GO"
        reason = "zero verified play-specific defensive executions"
    elif n < 10:
        verdict = "NO-GO"
        reason = "prior-driven sample; heuristic remains active"
    elif n < 40:
        verdict = "CONDITIONAL GO"
        reason = "limited verified sample; experimental only, hybrid refused"
    else:
        verdict = "CONDITIONAL GO"
        reason = "enough snaps for hybrid consideration after held-out eval"
    return {
        "verdict": verdict,
        "supervised_play_specific": n,
        "verified_human_games": games,
        "reason": reason,
        "auto_live_on_enable": False,
    }
