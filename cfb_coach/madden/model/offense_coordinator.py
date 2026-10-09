"""Joint formation/play/adjustment decision search for experimental offense."""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.offense_action_learning import load_action_evidence
from cfb_coach.madden.model.offense_action_policy import choose_offense_action


POLICY_VERSION = "adaptive_offensive_coordinator.joint.v1"
MAX_ACTION_INFLUENCE = 0.04


def _bounded_action_value(plan: Mapping[str, Any], no_action_score: float) -> float:
    if plan.get("kind") == "none":
        return 0.0
    raw = 0.35 * (float(plan.get("score") or 0.0) - no_action_score)
    return max(-MAX_ACTION_INFLUENCE, min(MAX_ACTION_INFLUENCE, raw))


def _decision_for_plan(
    action_search: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    kind = str(plan.get("kind") or "none")
    payload = plan.get("payload")
    is_none = kind == "none"
    return {
        "policy_version": POLICY_VERSION,
        "kind": "none" if is_none else kind,
        "id": None if is_none else plan.get("id"),
        "macro": payload if kind == "macro" else None,
        "adjustment": payload if kind in ("adjustment", "multi_adjustment") else None,
        "reason": str(plan.get("why") or ""),
        "scores_are": (
            "joint_play_probability_plus_explicit_football_proxy_and_"
            "conservative_research_action_prior"
        ),
        "candidates": action_search.get("candidates") or [],
        "legal_plan_count": len(action_search.get("legal_plans") or []),
        "no_action_compared": True,
        "no_action_score": action_search.get("no_action_score"),
        "top_action_score": plan.get("score"),
        "observational_evidence_mode": action_search.get(
            "observational_evidence_mode", "none"
        ),
        "observational_model_shift": plan.get("observational_model_shift", 0.0),
        "observational_model_n": plan.get("observational_model_n"),
        "observation_not_causal": True,
        "complexity": int(plan.get("complexity") or 0),
        "play_clock_feasible": bool(plan.get("play_clock_feasible", True)),
        "why_now": (
            "play unchanged won joint search"
            if is_none else "legal adjustment plan improved the joint proxy"
        ),
    }


def choose_joint_offensive_decision(
    ranked_plays: Sequence[Mapping[str, Any]],
    *,
    sit: Any,
    book: dict[str, list[str]],
    active: Sequence[str],
    db: Any,
    opponent_id: str,
    weights: Mapping[str, float] | None = None,
    cooled: set[str] | None = None,
    score_phase: str | None = None,
    audibles: dict[str, list[str]] | None = None,
    allow_macros: bool = True,
    repeated: bool = False,
    session_id: str | None = None,
    snap_seq: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Score every eligible installed play together with every legal plan.

    Candidate construction is exhaustive over the situation-eligible play
    list and bounded to zero, one, or two sourced adjustment operations.
    """
    if not ranked_plays:
        raise ValueError("joint offensive search requires eligible ranked plays")
    learned = load_action_evidence(db) if db is not None else None
    joint: list[dict[str, Any]] = []
    action_searches: dict[tuple[str, str], dict[str, Any]] = {}
    for play_row in ranked_plays:
        formation = str(play_row["formation"])
        play = str(play_row["play"])
        search = choose_offense_action(
            formation=formation,
            play=play,
            sit=sit,
            book=book,
            active=active,
            weights=weights,
            cooled=cooled,
            score_phase=score_phase,
            audibles=audibles,
            prediction=play_row,
            allow_macros=allow_macros,
            repeated=repeated,
            db=db,
            opponent_id=opponent_id,
            action_evidence=learned,
        )
        action_searches[(formation, play)] = search
        no_score = float(search.get("no_action_score") or 0.0)
        plans = list(search.get("legal_plans") or [])
        if not any(p.get("kind") == "none" for p in plans):
            plans.append({
                "kind": "none", "id": "NO_ADJUSTMENT", "score": no_score,
                "payload": None, "why": "run the selected play unchanged",
                "complexity": 0, "play_clock_feasible": True,
            })
        for plan in plans:
            if not plan.get("play_clock_feasible", True):
                continue
            action_value = _bounded_action_value(plan, no_score)
            base = float(play_row.get("selection_score", play_row["probability"]))
            joint.append({
                "formation": formation,
                "play": play,
                "play_row": dict(play_row),
                "action_plan": dict(plan),
                "play_score": round(base, 7),
                "action_value": round(action_value, 7),
                "joint_score": round(base + action_value, 7),
            })
    if not joint:
        raise ValueError("joint offensive search produced no legal decisions")

    joint.sort(key=lambda r: (
        -float(r["joint_score"]),
        int((r["action_plan"] or {}).get("complexity") or 0),
        r["formation"], r["play"], str((r["action_plan"] or {}).get("id")),
    ))
    ceiling = float(joint[0]["joint_score"])
    competitive = [row for row in joint if float(row["joint_score"]) >= ceiling - 0.055]
    seed = (
        f"{session_id or 'unscoped'}:{snap_seq if snap_seq is not None else 0}:"
        f"{getattr(sit, 'down', None)}:{getattr(sit, 'distance', None)}:joint"
    )
    rng = random.Random(int.from_bytes(
        hashlib.sha256(seed.encode("utf-8")).digest()[:8], "big"
    ))
    weights_for_sample = [
        math.exp(max(-12.0, (float(row["joint_score"]) - ceiling) / 0.035))
        for row in competitive
    ]
    chosen = rng.choices(competitive, weights=weights_for_sample, k=1)[0]
    search = action_searches[(chosen["formation"], chosen["play"])]
    decision = _decision_for_plan(search, chosen["action_plan"])
    audit = {
        "policy": POLICY_VERSION,
        "eligible_play_count": len(ranked_plays),
        "legal_joint_decision_count": len(joint),
        "plays_with_action_search": len(action_searches),
        "competitive_joint_count": len(competitive),
        "no_adjustment_candidates": sum(
            row["action_plan"].get("kind") == "none" for row in joint
        ),
        "multi_adjustment_candidates": sum(
            row["action_plan"].get("kind") == "multi_adjustment" for row in joint
        ),
        "selected": {
            "formation": chosen["formation"],
            "play": chosen["play"],
            "action_kind": chosen["action_plan"].get("kind"),
            "action_id": chosen["action_plan"].get("id"),
            "play_score": chosen["play_score"],
            "action_value": chosen["action_value"],
            "joint_score": chosen["joint_score"],
        },
        "action_influence_bound": MAX_ACTION_INFLUENCE,
        "seed_source": "stable_session_snap",
        "scores_are": (
            "learned_play_baseline_plus_interpretable_situation_proxy; "
            "adjustments use conservative research priors and gated observational tie-breaks"
        ),
        "top_alternatives": [
            {
                "formation": row["formation"], "play": row["play"],
                "action_kind": row["action_plan"].get("kind"),
                "action_id": row["action_plan"].get("id"),
                "joint_score": row["joint_score"],
            }
            for row in joint[:12]
        ],
    }
    return dict(chosen["play_row"]), decision, audit


__all__ = [
    "MAX_ACTION_INFLUENCE",
    "POLICY_VERSION",
    "choose_joint_offensive_decision",
]
