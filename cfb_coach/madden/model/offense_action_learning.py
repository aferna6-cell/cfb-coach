"""Conservative observational learner for executed Madden offensive adjustments.

Only exact, verified play/action executions with usable outcomes are eligible.
A displayed hot route or unchecked checkbox is NEVER evidence of application.
Comparisons are descriptive correlations, not causal adjustment effects.
Live model influence requires explicit enable AND held-out evidence gates.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping

from cfb_coach.madden.model import dataset, experimental_model

META_MODEL = "ml_offense_action_evidence.v1"
META_MODE = "ml_offense_action_evidence_mode.v1"
MIN_ACTION = 12
MIN_REFERENCE = 12
MIN_GAMES = 3
PRIOR_STRENGTH = 18.0
MAX_SCORE_INFLUENCE = 0.035


def _parse(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        result = json.loads(value or "{}")
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def _context(play: str, down: Any, distance: Any) -> str:
    return (
        experimental_model._play_concept(play) + "|" +
        experimental_model._dd_bucket(down, distance)
    )


def collect_action_evidence(db: Any) -> dict[str, Any]:
    """Join sealed model decision, verified outcome, and labeled executed snap.

    No cross-game joins or synthetic action negatives. No outcomes are
    credited to a suggested play if the user executed a different play.
    """
    by_snap: dict[str, dict[str, Any]] = {}
    for row in dataset.build_rows(db=db):
        if (row.get("snap_id")
                and row.get("eligibility") == dataset.ELIGIBILITY_VERIFIED_EXECUTION
                and row.get("success") in ("true", "false")):
            by_snap[str(row["snap_id"])] = row

    decisions: dict[str, Any] = {}
    outcomes: dict[str, Any] = {}
    try:
        for item in db.conn.execute("SELECT * FROM ml_decisions ORDER BY id"):
            decisions[str(item["snap_id"])] = item
        for item in db.conn.execute("SELECT * FROM ml_outcomes ORDER BY id"):
            outcomes[str(item["snap_id"])] = item
    except Exception:  # noqa: BLE001 — schema may not yet be migrated
        return {"rows": [], "counts": {"schema_unavailable": 1}}

    excluded: dict[str, int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    for snap_id, dec in decisions.items():
        snap = by_snap.get(snap_id)
        out = outcomes.get(snap_id)
        if snap is None or out is None:
            excluded["missing_verified_labeled_outcome"] += 1
            continue
        if out["executed_status"] != "identified" or out["executed_verification"] != "verified":
            excluded["unverified_execution"] += 1
            continue
        if (not snap.get("game_id") or
                str(snap.get("executed_formation")) != str(dec["final_formation"]) or
                str(snap.get("executed_play")) != str(dec["final_play"])):
            excluded["different_play_or_missing_game"] += 1
            continue
        action = (
            _parse(dec["decision_json"]).get("experimental_offense") or {}
        ).get("offense_action") or {}
        proposed_kind = str(action.get("kind") or "none")
        proposed_id = str(action.get("id") or "")
        result = _parse(out["outcome_json"])
        confirmed = result.get("offense_action_explicitly_confirmed") is True
        actual_macro = result.get("executed_macro")
        actual_adj = result.get("executed_adjustment_id")
        if proposed_kind in ("macro", "adjustment") and proposed_id:
            matched = (
                confirmed and
                ((proposed_kind == "macro" and actual_macro == proposed_id and not actual_adj)
                 or (proposed_kind == "adjustment" and actual_adj == proposed_id and not actual_macro))
            )
            if not matched:
                excluded["suggested_action_not_explicitly_verified"] += 1
                continue
            label = f"{proposed_kind}:{proposed_id}"
        elif proposed_kind == "none" and not proposed_id:
            # This is only an observational reference: no action was suggested
            # and none was *recorded*, NOT proof that no untracked adjustment ran.
            if confirmed or actual_macro or actual_adj:
                excluded["unattributed_action"] += 1
                continue
            label = "reference:no_action_recommended"
        else:
            excluded["incomplete_action_proposal"] += 1
            continue
        records.append({
            "snap_id": snap_id, "game_id": str(snap["game_id"]),
            "action": label,
            "context": _context(str(snap["executed_play"]), snap["down"], snap["distance"]),
            "success": snap["success"] == "true",
            "formation": snap["executed_formation"], "play": snap["executed_play"],
        })
    return {
        "rows": records,
        "counts": {
            "verified_action_applications": sum(
                not r["action"].startswith("reference:") for r in records
            ),
            "observed_no_recommended_action_references": sum(
                r["action"].startswith("reference:") for r in records
            ),
            "excluded": dict(excluded),
        },
    }


def fit_action_model(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Shrink observational action success toward comparable reference snaps."""
    examples = list(evidence.get("rows") or [])
    groups: dict[str, dict[str, Any]] = {}
    for item in examples:
        key = f"{item['context']}|{item['action']}"
        bucket = groups.setdefault(key, {
            "context": item["context"], "action": item["action"],
            "n": 0, "success": 0, "games": set(),
        })
        bucket["n"] += 1
        bucket["success"] += int(item["success"])
        bucket["games"].add(item["game_id"])

    references = {b["context"]: b for b in groups.values()
                  if b["action"] == "reference:no_action_recommended"}
    evaluated = []
    for bucket in sorted(groups.values(), key=lambda v: (v["context"], v["action"])):
        if bucket["action"].startswith("reference:"):
            continue
        baseline = references.get(bucket["context"])
        games_a = bucket["games"]
        games_b = baseline["games"] if baseline else set()
        n_b = baseline["n"] if baseline else 0
        can_use = (
            bucket["n"] >= MIN_ACTION and n_b >= MIN_REFERENCE
            and len(games_a) >= MIN_GAMES and len(games_b) >= MIN_GAMES
        )
        baseline_rate = (
            (baseline["success"] + 0.5 * PRIOR_STRENGTH)
            / (n_b + PRIOR_STRENGTH)
            if baseline else .5
        )
        act_rate = (
            bucket["success"] + baseline_rate * PRIOR_STRENGTH
        ) / (bucket["n"] + PRIOR_STRENGTH)
        delta = act_rate - baseline_rate
        evaluated.append({
            "context": bucket["context"], "action": bucket["action"],
            "verified_action_n": bucket["n"],
            "action_success_n": bucket["success"],
            "action_games": len(games_a),
            "reference_n": n_b,
            "reference_games": len(games_b),
            "observed_action_success_rate_shrunk": round(act_rate, 5),
            "observed_reference_success_rate_shrunk": round(baseline_rate, 5),
            "observed_difference_not_causal": round(delta, 5),
            "eligible_for_tiny_rank_adjustment": can_use,
            # Limit noisy action comparisons to a tiny contextual tie-breaker.
            "ranking_shift": round(
                max(-MAX_SCORE_INFLUENCE, min(MAX_SCORE_INFLUENCE, 0.3 * delta))
                if can_use else 0.0, 5
            ),
        })

    return {
        "schema": "madden.offense.action_evidence.v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": "verified_matching_play_and_explicitly_confirmed_action_only",
        "examples": len(examples),
        "counts": evidence.get("counts") or {},
        "entries": evaluated,
        "eligible_entries": sum(x["eligible_for_tiny_rank_adjustment"] for x in evaluated),
        "default_mode": "shadow",
        "limitation": (
            "Observational, not randomized or causal. Sparse estimates cannot "
            "justify claims that an adjustment causes higher success; verified "
            "and comparable game coverage are required before tiny influence."
        ),
    }


def train_from_db(db: Any) -> dict[str, Any]:
    return fit_action_model(collect_action_evidence(db))


def save_artifact(db: Any, artifact: Mapping[str, Any]) -> None:
    db.set_meta(META_MODEL, json.dumps(artifact, sort_keys=True))
    if db.get_meta(META_MODE) not in ("shadow", "enabled"):
        db.set_meta(META_MODE, "shadow")


def set_mode(db: Any, mode: str) -> dict[str, Any]:
    if mode not in ("shadow", "enabled"):
        raise ValueError("Action learning mode must be shadow or enabled")
    model = _parse(db.get_meta(META_MODEL))
    if mode == "enabled" and not model.get("eligible_entries"):
        raise ValueError(
            "No action has 12 verified applications, 12 comparable references "
            "and at least 3 distinct games in each group. Stay in shadow mode."
        )
    db.set_meta(META_MODE, mode)
    return {"mode": mode, "eligible_entries": model.get("eligible_entries", 0)}


def action_signal(db: Any, *, kind: str, action_id: str,
                  play: str, down: Any, distance: Any) -> dict[str, Any]:
    """Read-only and bounded; no stored model => zero influence."""
    empty = {
        "mode": "shadow", "model_available": False,
        "qualified": False, "ranking_shift": 0.0,
        "observational_not_causal": True,
    }
    if db is None or not kind or not action_id:
        return empty
    try:
        model = _parse(db.get_meta(META_MODEL))
        mode = db.get_meta(META_MODE) or "shadow"
        context = _context(play, down, distance)
        match = next(
            (v for v in model.get("entries") or []
             if v.get("context") == context
             and v.get("action") == f"{kind}:{action_id}"), None
        )
        if not match:
            return {**empty, "model_available": bool(model), "mode": mode}
        qualified = match.get("eligible_for_tiny_rank_adjustment") is True
        return {
            "mode": mode, "model_available": True, "qualified": qualified,
            "verified_n": match["verified_action_n"],
            "reference_n": match["reference_n"],
            "ranking_shift": (
                float(match["ranking_shift"])
                if mode == "enabled" and qualified else 0.0
            ),
            "shadow_ranking_shift": (
                float(match["ranking_shift"]) if qualified else 0.0
            ),
            "observational_not_causal": True,
        }
    except Exception:  # noqa: BLE001
        return empty
