"""Per-game audit of candidate defensive training rows.

The reported "178 defensive snaps" (and any similar aggregate) are candidates
for auditing, not automatically valid supervised examples. This module reports
verified counts by game and explains why uncertain executions are excluded
from play-specific training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.defense_verified import (
    VERIFIED_DEFINITION,
    partition_verified,
    verification_source,
    verified_counts,
)

AUDIT_SCHEMA = "madden.defense.training_audit.v1"


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _game_id(row: Mapping[str, Any]) -> str:
    for key in ("game_id", "match_id", "session_id", "vod_id"):
        if not _blank(row.get(key)):
            return str(row.get(key))
    return "unknown"


def _is_defense(row: Mapping[str, Any]) -> bool:
    side = str(row.get("side") or "").strip().lower()
    return side in ("", "defense", "d") or side.startswith("defense")


def exclusion_reason(row: Mapping[str, Any]) -> str | None:
    """Why a defensive candidate is not a verified supervised example."""
    if not isinstance(row, Mapping):
        return "not_a_mapping"
    if row.get("exclude") or row.get("invalid"):
        return "explicitly_excluded"
    if not _is_defense(row):
        return "not_defense"
    if verification_source(row):
        return None
    if row.get("recommendation_only") is True:
        return "recommendation_only"
    from cfb_coach.madden.model.dataset import vod_source_excluded

    if vod_source_excluded(row):
        return "vod_source_excluded"
    if row.get("trusted_vod") or row.get("vod_id") or row.get("observed_defense"):
        concept = str(row.get("concept_seen") or "").strip()
        if not concept:
            return "vod_missing_concept_seen"
        return "vod_not_reviewed_or_permitted"
    verification = str(row.get("executed_verification") or "").strip().lower()
    status = str(row.get("executed_status") or "").strip().lower()
    accepted = str(row.get("accepted") or "").strip().lower()
    if _blank(verification) and _blank(status) and _blank(accepted):
        return "execution_unrecorded"
    if verification and verification != "verified":
        return f"executed_verification={verification or 'empty'}"
    if status and status not in {"identified", "accepted", "executed"}:
        return f"executed_status={status}"
    if accepted and accepted not in {
        "accepted", "executed", "used_recommended", "yes", "true",
    }:
        return f"accepted={accepted}"
    return "not_verified_execution"


def audit_defensive_training(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Supervised counts by game plus exclusion tallies.

    Uncertain executions stay out of play-specific training counts. Opponent
    ids are reported only as labels; John is not privileged.
    """
    candidates = [dict(row) for row in rows if _is_defense(row)]
    parts = partition_verified(candidates)
    verified = parts["live"] + parts["vod"]
    by_game: dict[str, dict[str, Any]] = {}
    for row in verified:
        gid = _game_id(row)
        slot = by_game.setdefault(
            gid,
            {
                "game_id": gid,
                "verified_live": 0,
                "verified_vod": 0,
                "verified_total": 0,
                "opponents": set(),
                "with_success_label": 0,
                "with_yards": 0,
            },
        )
        source = verification_source(row)
        if source == "live":
            slot["verified_live"] += 1
        elif source == "vod":
            slot["verified_vod"] += 1
        slot["verified_total"] += 1
        if not _blank(row.get("opponent_id")):
            slot["opponents"].add(str(row.get("opponent_id")))
        success = row.get("success")
        if success is None:
            success = row.get("stop")
        if isinstance(success, bool) or (success is not None and str(success).strip() != ""):
            slot["with_success_label"] += 1
        try:
            int(row["yards"])
            slot["with_yards"] += 1
        except (KeyError, TypeError, ValueError):
            pass

    games = []
    for gid in sorted(by_game):
        slot = by_game[gid]
        games.append({
            "game_id": slot["game_id"],
            "verified_live": slot["verified_live"],
            "verified_vod": slot["verified_vod"],
            "verified_total": slot["verified_total"],
            "opponents": sorted(slot["opponents"]),
            "with_success_label": slot["with_success_label"],
            "with_yards": slot["with_yards"],
            "play_specific_trainable": slot["with_success_label"],
        })

    exclusion_counts: Counter[str] = Counter()
    excluded_examples: list[dict[str, Any]] = []
    for row in candidates:
        reason = exclusion_reason(row)
        if reason is None:
            continue
        exclusion_counts[reason] += 1
        if len(excluded_examples) < 12:
            excluded_examples.append({
                "game_id": _game_id(row),
                "reason": reason,
                "play": row.get("executed_play") or row.get("play") or row.get("action_play"),
            })

    counts = verified_counts(candidates)

    def _has_label(row: Mapping[str, Any]) -> bool:
        success = row.get("success")
        if success is None:
            success = row.get("stop")
        if isinstance(success, bool):
            return True
        return not _blank(success)

    play_specific = sum(1 for row in verified if _has_label(row))
    return {
        "schema": AUDIT_SCHEMA,
        "verified_definition": VERIFIED_DEFINITION,
        "candidate_defensive_rows": len(candidates),
        "verified_by_source": {
            "live": counts["live"],
            "vod": counts["vod"],
            "total": counts["total"],
        },
        "play_specific_supervised": play_specific,
        "excluded_from_play_specific": len(candidates) - counts["total"],
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "exclusion_examples": excluded_examples,
        "verified_by_game": games,
        "games_with_verified": len(games),
        "note": (
            "Candidate aggregates (including any reported 178) are not "
            "automatically supervised examples. Uncertain executions are "
            "excluded from play-specific training."
        ),
        "promotion_ready": False,
        "auto_promote": False,
    }


def format_audit(report: Mapping[str, Any]) -> str:
    lines = [
        f"schema: {report.get('schema')}",
        f"verified_definition: {report.get('verified_definition')}",
        f"candidate_defensive_rows: {report.get('candidate_defensive_rows')}",
        (
            "verified_live/vod/total: "
            f"{report.get('verified_by_source', {}).get('live', 0)}/"
            f"{report.get('verified_by_source', {}).get('vod', 0)}/"
            f"{report.get('verified_by_source', {}).get('total', 0)}"
        ),
        f"play_specific_supervised: {report.get('play_specific_supervised')}",
        f"excluded_from_play_specific: {report.get('excluded_from_play_specific')}",
        f"games_with_verified: {report.get('games_with_verified')}",
    ]
    for game in report.get("verified_by_game") or []:
        lines.append(
            f"  game {game['game_id']}: verified={game['verified_total']} "
            f"(live={game['verified_live']} vod={game['verified_vod']}) "
            f"trainable_labels={game['play_specific_trainable']} "
            f"opponents={','.join(game['opponents']) or '-'}"
        )
    exclusions = report.get("exclusion_counts") or {}
    if exclusions:
        lines.append("exclusion_counts:")
        for reason, count in exclusions.items():
            lines.append(f"  {reason}: {count}")
    lines.append(f"note: {report.get('note')}")
    lines.append("auto_promote: false")
    return "\n".join(lines)
