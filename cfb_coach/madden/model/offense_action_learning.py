"""Evidence-gated observational learning for Madden offensive actions.

An action recommendation is never an applied action. A checked, verified
execution AND labeled outcome are required. The unchanged-play comparator
is admissible only with a separate explicit no-adjustment confirmation.

This is observational association, not causal uplift: we use aggressive
shrinkage, per-context comparator and distinct-game gates before optionally
influencing live selection. Research/eligibility rules always take precedence.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.experimental_model import _play_concept, _dd_bucket

META_ACTION_MODEL = "ml_offense_action_evidence.v1"
SCHEMA = "madden.offense.action_evidence.v1"
MIN_ACTION = 12
MIN_CONTROL = 12
MIN_GAMES = 3
PRIOR_STRENGTH = 18.0
MAX_SCORE_SHIFT = 0.04


def context_key(*, play: str, down: Any, distance: Any) -> str:
    """Pre-snap context only (no opponent post-snap coverage leakage)."""
    return f"{_play_concept(play)}|{_dd_bucket(down, distance)}"


def collect_verified_action_rows(db: Any, game_id: str | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Join sealed decisions to active outcomes; deduplicate by snap ID."""
    from cfb_coach.madden.model.dataset import build_rows

    records = build_rows(db=db)
    decision_map: dict[str, dict[str, Any]] = {}
    outcome_map: dict[str, dict[str, Any]] = {}
    for row in db.conn.execute("SELECT * FROM ml_decisions WHERE snap_id IS NOT NULL ORDER BY id"):
        decision_map[str(row["snap_id"])] = dict(row)
    for row in db.conn.execute("SELECT * FROM ml_outcomes WHERE snap_id IS NOT NULL ORDER BY id"):
        outcome_map[str(row["snap_id"])] = dict(row)
    admitted = []
    stats = {
        "rows_seen": 0, "verified_labeled_play": 0,
        "verified_applied_actions": 0, "verified_unchanged": 0,
        "excluded_no_action_confirmation": 0,
        "excluded_action_confirmation": 0,
        "excluded_play_mismatch": 0,
    }
    seen: set[str] = set()
    for row in records:
        if row.get("side") != "offense":
            continue
        sid = str(row.get("snap_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        decision = decision_map.get(sid)
        outcome = outcome_map.get(sid)
        if not decision or not outcome:
            continue
        if game_id and str(decision.get("game_id") or "") != game_id:
            continue
        stats["rows_seen"] += 1
        if not row.get("supervised_eligible") or str(row.get("success")) not in ("true", "false"):
            continue
        if str(outcome.get("executed_verification") or "") != "verified":
            continue
        if (outcome.get("executed_formation") != decision.get("final_formation")
                or outcome.get("executed_play") != decision.get("final_play")):
            stats["excluded_play_mismatch"] += 1
            continue
        stats["verified_labeled_play"] += 1
        try:
            choice = (json.loads(decision.get("decision_json") or "{}")
                      .get("experimental_offense") or {}).get("offense_action") or {}
            result = json.loads(outcome.get("outcome_json") or "{}")
        except (ValueError, TypeError):
            continue
        kind = str(choice.get("kind") or "none")
        action_id = str(choice.get("id") or "") if kind in ("macro", "adjustment") else ""
        explicitly_applied = (
            kind in ("macro", "adjustment")
            and bool(result.get("offense_action_explicitly_confirmed"))
            and (
                kind == "macro" and result.get("executed_macro") == action_id
                or kind == "adjustment" and result.get("executed_adjustment_id") == action_id
            )
        )
        explicitly_unchanged = (
            kind == "none" and result.get("no_adjustment_explicitly_confirmed") is True
            and not result.get("offense_action_explicitly_confirmed")
            and not result.get("executed_macro")
            and not result.get("executed_adjustment_id")
        )
        if not explicitly_applied and not explicitly_unchanged:
            stats["excluded_action_confirmation" if action_id else "excluded_no_action_confirmation"] += 1
            continue
        if explicitly_applied:
            stats["verified_applied_actions"] += 1
        else:
            stats["verified_unchanged"] += 1
        admitted.append({
            "snap_id": sid,
            "game_id": str(decision.get("game_id") or "unknown"),
            "formation": str(decision["final_formation"]),
            "play": str(decision["final_play"]),
            "context": context_key(
                play=str(decision["final_play"]),
                down=row.get("down"), distance=row.get("distance"),
            ),
            "action": f"{kind}:{action_id}" if explicitly_applied else "none",
            "kind": kind if explicitly_applied else "none",
            "success": str(row["success"]) == "true",
            "evidence": "verified_executed_play_plus_explicit_action_outcome",
        })
    return admitted, stats


def fit_action_evidence(
    rows: Sequence[Mapping[str, Any]], *, stats: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Shrink within-context observational rates and gate action promotion."""
    counts: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(lambda: {"success": 0, "total": 0, "games": set()})
    )
    for row in rows:
        ctx = str(row["context"])
        action = str(row["action"])
        cell = counts[ctx][action]
        cell["total"] += 1
        cell["success"] += int(bool(row["success"]))
        cell["games"].add(str(row["game_id"]))
    comparisons: dict[str, dict[str, Any]] = {}
    total_ready = 0
    for ctx, options in sorted(counts.items()):
        control = options.get("none", {"success": 0, "total": 0, "games": set()})
        base_rate = (
            (float(control["success"]) + PRIOR_STRENGTH * .5)
            / (float(control["total"]) + PRIOR_STRENGTH)
        )
        for name, bucket in sorted(options.items()):
            if name == "none":
                continue
            n_action = int(bucket["total"])
            n_control = int(control["total"])
            posterior_action = (
                (float(bucket["success"]) + PRIOR_STRENGTH * base_rate)
                / (n_action + PRIOR_STRENGTH)
            )
            difference = posterior_action - base_rate
            games = len(bucket["games"] | control["games"])
            meets = (
                n_action >= MIN_ACTION and n_control >= MIN_CONTROL
                and len(bucket["games"]) >= MIN_GAMES
                and len(control["games"]) >= MIN_GAMES
            )
            # Only a consistent, sufficiently large associational signal
            # gets a tiny effect on live score when explicitly promoted.
            guarded = meets and abs(difference) >= .08
            shift = max(-MAX_SCORE_SHIFT, min(MAX_SCORE_SHIFT, difference * .20)) if guarded else 0.
            if guarded:
                total_ready += 1
            comparisons[f"{ctx}|{name}"] = {
                "context": ctx, "action": name,
                "n_action": n_action, "n_unchanged": n_control,
                "action_success": int(bucket["success"]),
                "unchanged_success": int(control["success"]),
                "games_action": len(bucket["games"]),
                "games_unchanged": len(control["games"]),
                "games_total": games,
                "posterior_association": round(difference, 6),
                "meets_minimums": meets, "ready_for_bounded_adjustment": guarded,
                "bounded_selection_shift": round(shift, 6),
                "caveat": "observational association, confounding not removed",
            }
    digest = hashlib.sha256(json.dumps(
        sorted((r["snap_id"], r["action"], r["success"]) for r in rows),
        sort_keys=True, default=str,
    ).encode("utf-8")).hexdigest()[:20]
    return {
        "schema": SCHEMA, "mode": "shadow", "fingerprint": digest,
        "trained_ts": datetime.now(timezone.utc).isoformat(),
        "n_verified_action_rows": len(rows),
        "n_context_action_groups": len(comparisons),
        "ready_groups": total_ready, "stats": dict(stats or {}),
        "comparisons": comparisons,
        "bounds": {
            "min_action": MIN_ACTION, "min_unchanged": MIN_CONTROL,
            "min_distinct_games_each": MIN_GAMES,
            "max_selection_score_shift": MAX_SCORE_SHIFT,
        },
        "note": (
            "Noncausal observational comparisons only. Never infer adjustment "
            "application or unmodified execution from a displayed recommendation. "
            "Insufficient evidence -> zero selection-score adjustment."
        ),
    }


def train_action_evidence(db: Any, game_id: str | None = None) -> dict[str, Any]:
    rows, stats = collect_verified_action_rows(db, game_id=game_id)
    return fit_action_evidence(rows, stats=stats)


def load_action_evidence(db: Any) -> dict[str, Any] | None:
    raw = db.get_meta(META_ACTION_MODEL) if db is not None else None
    if not raw:
        return None
    try:
        item = json.loads(raw)
        if item.get("schema") == SCHEMA:
            return item
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def save_action_evidence(db: Any, artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema") != SCHEMA:
        raise ValueError("Unknown action artifact schema")
    db.set_meta(META_ACTION_MODEL, json.dumps(dict(artifact), sort_keys=True))


def promote_action_evidence(db: Any) -> dict[str, Any]:
    art = load_action_evidence(db)
    if not art:
        raise ValueError("Train the action evidence model before attempting promotion")
    if int(art.get("ready_groups") or 0) < 1:
        raise ValueError("No action has enough verified applied AND unchanged comparisons across games")
    promoted = dict(art, mode="bounded_active")
    save_action_evidence(db, promoted)
    return promoted


def rollback_action_evidence(db: Any) -> dict[str, Any]:
    art = load_action_evidence(db)
    if not art:
        return {"mode": "none"}
    shadow = dict(art, mode="shadow")
    save_action_evidence(db, shadow)
    return shadow


def score_shift(
    artifact: Mapping[str, Any] | None, *,
    play: str, down: Any, distance: Any, kind: str, action_id: str,
) -> tuple[float, dict[str, Any] | None]:
    """Read-only bounded adjustment only for promoted, adequately compared context."""
    if not artifact or artifact.get("mode") != "bounded_active":
        return 0.0, None
    key = f"{context_key(play=play, down=down, distance=distance)}|{kind}:{action_id}"
    group = (artifact.get("comparisons") or {}).get(key)
    if not group or not group.get("ready_for_bounded_adjustment"):
        return 0.0, None
    return float(group.get("bounded_selection_shift") or 0.0), group
