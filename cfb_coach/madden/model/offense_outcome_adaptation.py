"""Outcome-backed, bounded play penalties for Madden 27 offense.

Only *identified, user-verified executions with observed outcomes* can alter a
live selection score. Recommendations, unverified snaps, and hypothetical calls
do not count. These are shrinkage penalties, not counterfactual win estimates.
No clock, live coverage, or manually identified pre-snap defense is required.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

MAX_TOTAL_PENALTY = 0.105
GOAL_LINE_MAX = 0.075
SCREEN_MAX = 0.070
RECENT_FAILURE_MAX = 0.048


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cell(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def verified_history(
    db: Any, opponent_id: str, *, limit: int = 240,
    before_snap_id: int | None = None,
) -> list[dict[str, Any]]:
    """Newest first. Bounded query; no future snap leakage in offline replay."""
    if db is None or not opponent_id:
        return []
    try:
        sql = (
            "SELECT id, session_id, opponent_id, executed_formation, executed_play,"
            " result, yards, down, distance, yardline, executed_status,"
            " executed_verification FROM snaps "
            "WHERE opponent_id=? AND side='offense' AND "
            "executed_status='identified' AND executed_verification='verified' "
            "AND executed_play IS NOT NULL AND result IS NOT NULL"
        )
        args: list[Any] = [opponent_id]
        if before_snap_id is not None:
            sql += " AND id < ?"
            args.append(int(before_snap_id))
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(max(1, min(int(limit), 600)))
        return [dict(row) for row in db.conn.execute(sql, args)]
    except Exception:  # noqa: BLE001 - optional learning cannot block a live pick
        return []


def _loss(row: Mapping[str, Any]) -> bool:
    outcome = str(row.get("result") or "").lower().strip()
    if outcome in ("int", "fumble lost", "sack", "incomplete", "stop"):
        return True
    yards = _as_int(row.get("yards"))
    if yards is not None:
        return yards < 0
    if outcome.startswith("-"):
        return True
    return False


def _goal_to_go(sit: Any) -> bool:
    yl = _as_int(getattr(sit, "yardline", None))
    return bool(getattr(sit, "goal_line", False) or (yl is not None and yl >= 95))


def outcome_penalty(
    form: str, play: str, sit: Any,
    history: Sequence[Mapping[str, Any]],
    *, session_id: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return bounded penalty for *this* formation/play and current context."""
    selected = [r for r in history if str(r.get("executed_play") or "") == play]
    exact = [r for r in selected if str(r.get("executed_formation") or "") == form]
    reasons: list[str] = []
    parts: dict[str, float] = {}

    if _goal_to_go(sit):
        goal = [
            r for r in selected
            if (_as_int(r.get("yardline")) or 0) >= 95
        ]
        goal_ints = sum(str(r.get("result") or "").lower() == "int" for r in goal)
        if len(goal) >= 2 and goal_ints:
            # Require repeated *same-play* failure near the goal line. Even
            # then, a pass remains available if its predicted advantage wins.
            parts["goal_line_interceptions"] = min(
                GOAL_LINE_MAX, 0.14 * goal_ints / (len(goal) + 2)
            )
            reasons.append(f"historical goal-to-go INT {goal_ints}/{len(goal)}")

    from cfb_coach.madden.catalog import is_run
    is_screen = "screen" in play.lower()
    if is_screen:
        recent = selected[:12]
        yard_values = [
            _as_int(r.get("yards")) for r in recent
            if _as_int(r.get("yards")) is not None
        ]
        losses = sum(_loss(r) for r in recent)
        if len(recent) >= 3 and yard_values:
            mean_yards = sum(yard_values) / len(yard_values)
            if mean_yards < 2.0 and losses >= 2:
                parts["recent_screen_outcomes"] = min(
                    SCREEN_MAX, 0.10 * (2.0 - mean_yards) / (len(recent) + 4)
                    + 0.018 * losses / max(1, len(recent))
                )
                reasons.append(f"recent screen {mean_yards:+.1f} yards/play")

    # Pure same-session response to consecutive observed failures. The model
    # can switch after two genuine outcomes, not after a fixed snap count.
    if session_id:
        session_rows = [
            r for r in exact if str(r.get("session_id")) == session_id
        ]
        if len(session_rows) >= 2 and _loss(session_rows[0]) and _loss(session_rows[1]):
            parts["two_recent_failures"] = min(
                RECENT_FAILURE_MAX, 0.03 +
                0.009 * min(2, sum(_loss(r) for r in session_rows[:4]) - 2)
            )
            reasons.append("two latest same-play executions failed")

    # No fixed play ban. A strong predicted advantage can overcome the cost.
    amount = min(MAX_TOTAL_PENALTY, sum(parts.values()))
    return round(amount, 6), {
        "penalties": {k: round(v, 6) for k, v in parts.items()},
        "reasons": reasons,
        "evidence_count": len(selected),
        "observed_only": True,
    }


def apply_outcome_adaptation(
    ranked: Sequence[Mapping[str, Any]], *,
    sit: Any, history: Sequence[Mapping[str, Any]],
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    adjusted = []
    affected = 0
    for row in ranked:
        new = dict(row)
        cost, explain = outcome_penalty(
            str(row["formation"]), str(row["play"]), sit, history,
            session_id=session_id,
        )
        original = float(new.get("selection_score", new.get("probability", 0.0)))
        new["selection_score"] = round(original - cost, 7)
        new["outcome_penalty"] = cost
        new["outcome_penalty_reason"] = explain["reasons"]
        if cost:
            affected += 1
        adjusted.append(new)
    adjusted.sort(key=lambda r: (-float(r["selection_score"]), -float(r["probability"]),
                                  r["formation"], r["play"]))
    return adjusted, {
        "policy": "verified_outcome_adaptation.v1",
        "history_count": len(history),
        "penalized_candidates": affected,
        "largest_penalty": max((r["outcome_penalty"] for r in adjusted), default=0.0),
        "no_clock_or_showing": True,
    }
