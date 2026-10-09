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


def context_key(
    *, play: str, down: Any, distance: Any,
    coverage_class: str | None = None, coverage_source: str | None = None,
    red_zone: bool = False, goal_line: bool = False,
) -> str:
    """Decision-time context; only an explicitly *live* look can distinguish coverage.

    Unknown/last-snap hints do not form new action-effect buckets, and the
    opponent's actual post-snap coverage must NEVER be substituted here.
    """
    base = f"{_play_concept(play)}|{_dd_bucket(down, distance)}"
    if goal_line:
        base += "|zone:goal_line"
    elif red_zone:
        base += "|zone:red_zone"
    if coverage_source == "live" and coverage_class:
        return f"{base}|look:{coverage_class}"
    return base


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
            record = (json.loads(decision.get("decision_json") or "{}")
                      .get("experimental_offense") or {})
            choice = record.get("offense_action") or {}
            presnap = record.get("pre_snap_action_context") or {}
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
            if kind in ("macro", "adjustment"):
                stats["excluded_action_confirmation"] += 1
            else:
                stats["excluded_no_action_confirmation"] += 1
            continue
        if explicitly_applied:
            stats["verified_applied_actions"] += 1
        if explicitly_unchanged:
            stats["verified_unchanged"] += 1
        admitted.append({
            "snap_id": sid,
            "game_id": str(decision.get("game_id") or row.get("game_id")),
            "play": str(decision.get("final_play") or row.get("play")),
            "formation": str(decision.get("final_formation") or row.get("formation")),
            "down": presnap.get("down", row.get("down")),
            "distance": presnap.get("distance", row.get("distance")),
            "red_zone": bool(presnap.get("red_zone", False)),
            "goal_line": bool(presnap.get("goal_line", False)),
            "coverage_class": (
                presnap.get("coverage_hint")
                if presnap.get("coverage_source") == "live"
                else None
            ),
            "coverage_source": presnap.get("coverage_source"),
            "kind": kind if explicitly_applied else "none",
            "action_id": action_id if explicitly_applied else "",
            "success": 1.0 if str(row.get("success")) == "true" else 0.0,
            "yards": float(row.get("yards") or 0.0),
        })
    return admitted, stats


def fit_action_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate shrunk, context-isolated action score deltas."""
    by_context_control: dict[str, list[float]] = defaultdict(list)
    by_context_action: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    games_by_action: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for r in rows:
        key = context_key(
            play=str(r["play"]), down=r["down"], distance=r["distance"],
            coverage_class=r.get("coverage_class"),
            coverage_source=r.get("coverage_source"),
            red_zone=bool(r.get("red_zone", False)),
            goal_line=bool(r.get("goal_line", False)),
        )
        if r["kind"] == "none":
            by_context_control[key].append(float(r["success"]))
        else:
            act_key = (key, str(r["kind"]), str(r["action_id"]))
            by_context_action[act_key].append(float(r["success"]))
            games_by_action[act_key].add(str(r["game_id"]))

    groups = []
    ready = 0
    for (ctx, kind, act_id), succs in sorted(by_context_action.items()):
        controls = by_context_control.get(ctx, [])
        games = len(games_by_action[(ctx, kind, act_id)])
        n_act, n_ctrl = len(succs), len(controls)
        mean_act = sum(succs) / n_act if n_act else 0.0
        mean_ctrl = sum(controls) / n_ctrl if n_ctrl else 0.0
        raw_delta = mean_act - mean_ctrl
        weight = n_act / (n_act + PRIOR_STRENGTH)
        shrunk_delta = raw_delta * weight
        bounded = max(-MAX_SCORE_SHIFT, min(MAX_SCORE_SHIFT, shrunk_delta))
        eligible = n_act >= MIN_ACTION and n_ctrl >= MIN_CONTROL and games >= MIN_GAMES
        if eligible:
            ready += 1
        groups.append({
            "context_key": ctx, "kind": kind, "action_id": act_id,
            "n_action": n_act, "n_unchanged": n_ctrl, "distinct_games": games,
            "action_success_rate": round(mean_act, 4),
            "unchanged_success_rate": round(mean_ctrl, 4),
            "raw_delta": round(raw_delta, 4),
            "shrinkage_weight": round(weight, 4),
            "score_shift": round(bounded, 5),
            "eligible_for_live_gate": eligible,
        })
    groups.sort(key=lambda g: (-g["n_action"], g["context_key"]))
    return {
        "groups": groups, "ready_groups": ready, "n_groups": len(groups),
    }


def train_action_evidence(db: Any, *, game_id: str | None = None) -> dict[str, Any]:
    rows, stats = collect_verified_action_rows(db, game_id=game_id)
    fitted = fit_action_evidence(rows)
    raw = json.dumps({
        "schema": SCHEMA, "stats": stats, "fitted": fitted,
    }, sort_keys=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return {
        "schema": SCHEMA, "trained_ts": datetime.now(timezone.utc).isoformat(),
        "evidence_id": digest, "mode": "shadow",
        "n_verified_action_rows": len(rows), "stats": stats,
        "ready_groups": fitted["ready_groups"], "groups": fitted["groups"],
        "max_score_shift": MAX_SCORE_SHIFT, "prior_strength": PRIOR_STRENGTH,
        "gates": {
            "min_action_samples": MIN_ACTION,
            "min_unchanged_samples": MIN_CONTROL,
            "min_distinct_games": MIN_GAMES,
        },
        "notes": [
            "Observational action-vs-unchanged comparison under sealed pre-snap context.",
            "Not causal proof; aggressive empirical Bayes shrinkage and distinct-game gates applied.",
            "Live score shifts are bounded to +/-0.04 and require explicit gated promotion.",
        ],
    }


def save_action_evidence(db: Any, artifact: Mapping[str, Any]) -> None:
    db.set_meta(META_ACTION_MODEL, json.dumps(artifact, sort_keys=True))


def load_action_evidence(db: Any) -> dict[str, Any] | None:
    raw = db.get_meta(META_ACTION_MODEL)
    try:
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return None


def promote_action_evidence(db: Any, *, force: bool = False) -> dict[str, Any]:
    current = load_action_evidence(db)
    if not current:
        raise ValueError("No trained action evidence artifact found; run train first")
    ready = int(current.get("ready_groups", 0))
    if ready < 1 and not force:
        raise ValueError("No action-context groups meet the multi-game, paired-sample threshold")
    promoted = dict(current)
    promoted["mode"] = "bounded_active"
    promoted["promoted_ts"] = datetime.now(timezone.utc).isoformat()
    save_action_evidence(db, promoted)
    return promoted


def rollback_action_evidence(db: Any) -> dict[str, Any]:
    current = load_action_evidence(db)
    if not current:
        raise ValueError("No action evidence artifact found to roll back")
    rolled = dict(current)
    rolled["mode"] = "shadow"
    rolled["rolled_back_ts"] = datetime.now(timezone.utc).isoformat()
    save_action_evidence(db, rolled)
    return rolled


def score_shift(
    artifact: Mapping[str, Any] | None,
    *,
    play: str,
    down: Any,
    distance: Any,
    kind: str,
    action_id: str,
    coverage_class: str | None = None,
    coverage_source: str | None = None,
    red_zone: bool = False,
    goal_line: bool = False,
) -> tuple[float, dict[str, Any] | None]:
    """Retrieve the bounded observational score shift for a candidate action."""
    if not artifact or artifact.get("mode") != "bounded_active":
        return 0.0, None
    key = context_key(
        play=play, down=down, distance=distance,
        coverage_class=coverage_class, coverage_source=coverage_source,
        red_zone=red_zone, goal_line=goal_line,
    )
    for g in artifact.get("groups", []):
        if (g["context_key"] == key and g["kind"] == kind and g["action_id"] == action_id
                and g.get("eligible_for_live_gate")):
            return float(g["score_shift"]), g
    return 0.0, None
