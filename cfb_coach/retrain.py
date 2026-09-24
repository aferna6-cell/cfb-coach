"""Smarter postgame retrain — grade plays vs coverage/look from yardage outcomes."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from cfb_coach.outcome import outcome_success, parse_outcome


def _snap_get(s: Any, key: str, default: Any = None) -> Any:
    if isinstance(s, dict):
        return s.get(key, default)
    try:
        return s[key]
    except (KeyError, IndexError, TypeError):
        return getattr(s, key, default)


def grade_play_vs_look(snaps: list[Any]) -> list[dict[str, Any]]:
    """Aggregate our play (formation+play) vs coverage_seen / concept_seen.

    Returns graded rows sorted by sample size, then avg yards.
    """
    buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for s in snaps:
        form = (_snap_get(s, "formation") or "?").strip() or "?"
        play = (_snap_get(s, "play") or "?").strip() or "?"
        side = (_snap_get(s, "side") or "offense").strip()
        look = (
            (_snap_get(s, "coverage_seen") or "").strip()
            or (_snap_get(s, "concept_seen") or "").strip()
            or "unknown"
        )
        result = _snap_get(s, "result")
        ok = outcome_success(result, side)
        if ok is None:
            continue
        parsed = parse_outcome(result)
        key = (side, form, play, look)
        row = buckets.setdefault(
            key,
            {
                "side": side,
                "formation": form,
                "play": play,
                "look": look,
                "n": 0,
                "successes": 0,
                "yards_sum": 0,
                "yards_n": 0,
            },
        )
        row["n"] += 1
        if ok:
            row["successes"] += 1
        if parsed.yards is not None:
            row["yards_sum"] += parsed.yards
            row["yards_n"] += 1

    out: list[dict[str, Any]] = []
    for row in buckets.values():
        n = row["n"]
        rate = row["successes"] / n if n else 0.0
        avg_y = (row["yards_sum"] / row["yards_n"]) if row["yards_n"] else None
        out.append(
            {
                "side": row["side"],
                "formation": row["formation"],
                "play": row["play"],
                "look": row["look"],
                "n": n,
                "successes": row["successes"],
                "success_rate": round(rate, 3),
                "avg_yards": None if avg_y is None else round(avg_y, 2),
                "label": f"{row['formation']} — {row['play']} vs {row['look']}",
            }
        )
    out.sort(key=lambda r: (-r["n"], -(r["avg_yards"] or 0), -r["success_rate"]))
    return out


def apply_play_vs_look_weights(
    db: Any,
    opponent_id: str,
    grades: list[dict[str, Any]],
    *,
    bump: float = 0.12,
    fail_bump: float = 0.08,
    min_n: int = 1,
) -> dict[str, float]:
    """Bump/demote gameplan keys from play-vs-look grades. Returns changelog."""
    changes: dict[str, float] = {}
    for g in grades:
        if g["n"] < min_n:
            continue
        side = g["side"]
        key = f"vs_look::{g['formation']}::{g['play']}::{g['look']}"
        rate = g["success_rate"]
        if rate >= 0.6:
            delta = bump * (1.0 + min(g["n"] - 1, 3) * 0.15)
        elif rate <= 0.4:
            delta = -fail_bump * (1.0 + min(g["n"] - 1, 3) * 0.15)
        else:
            # Mild nudge from yards when known
            ay = g.get("avg_yards")
            if ay is None:
                continue
            if side == "offense":
                delta = bump * 0.35 if ay >= 4 else (-fail_bump * 0.35 if ay <= 1 else 0.0)
            else:
                delta = bump * 0.35 if ay <= 2 else (-fail_bump * 0.35 if ay >= 6 else 0.0)
            if delta == 0.0:
                continue
        db.bump_gameplan_weight(opponent_id, side, key, delta)
        ck = f"{opponent_id}/{side}:{key}"
        changes[ck] = round(changes.get(ck, 0.0) + delta, 3)
    return changes


def format_grades_summary(grades: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    lines: list[str] = []
    for g in grades[:limit]:
        yd = "" if g["avg_yards"] is None else f", avg {g['avg_yards']:+.1f} yds"
        lines.append(
            f"  {g['label']}: {g['successes']}/{g['n']} success ({g['success_rate']:.0%}){yd}"
        )
    return lines


def retrain_from_snaps(
    db: Any,
    opponent_id: str,
    snaps: list[Any],
    *,
    also_learn: bool = True,
) -> dict[str, Any]:
    """Grade this batch, bump vs-look weights, optionally run base learn_from_snaps.

    Returns a dict suitable for HTML / CLI summary.
    """
    grades = grade_play_vs_look(snaps)
    vs_changes = apply_play_vs_look_weights(db, opponent_id, grades)
    base: dict[str, Any] = {}
    if also_learn:
        from cfb_coach.gameplan import learn_from_snaps

        ids = [int(_snap_get(s, "id") or 0) for s in snaps]
        since = (min(ids) - 1) if ids else None
        # Restrict learn_from_snaps by using since_id just below this batch
        base = learn_from_snaps(db, opponent_id, since_id=since, also_global=True)
        # Filter "snaps_considered" narrative to this batch size when possible
        base["snaps_considered"] = len(snaps)
    return {
        "opponent_id": opponent_id,
        "snaps": len(snaps),
        "grades": grades,
        "vs_look_changes": vs_changes,
        "base": base,
        "grade_lines": format_grades_summary(grades),
    }


__all__ = [
    "apply_play_vs_look_weights",
    "format_grades_summary",
    "grade_play_vs_look",
    "retrain_from_snaps",
]
