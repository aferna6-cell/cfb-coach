"""Bounded expert/personalization signal for the model-primary joint decision.

New learning artifacts default to shadow mode. Live influence requires explicit
user approval and evaluation gates. Expert footage cannot replace the installed
playbook, invent formations, or invent controller adjustments.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from cfb_coach.madden.model.expert_policy import load_expert_policy
from cfb_coach.madden.model.personalization import (
    META_PERSONALIZATION,
    blend_for_play,
    load_personalization,
    save_personalization,
)

META_EXPERT_SIGNAL = "ml_offense_expert_signal.v1"
SIGNAL_SCHEMA = "madden.expert.signal.v1"
# Smaller than opponent learning / research so model-primary scoring stays primary.
EXPERT_SIGNAL_CAP = 0.018
DEFAULT_MODE = "shadow"


def load_expert_signal(db: Any = None) -> dict[str, Any] | None:
    if db is None:
        return None
    try:
        raw = db.get_meta(META_EXPERT_SIGNAL)
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_expert_signal(db: Any, artifact: Mapping[str, Any]) -> None:
    if db is None:
        return
    db.set_meta(META_EXPERT_SIGNAL, json.dumps(dict(artifact), sort_keys=True))


def register_shadow_artifacts(
    db: Any,
    *,
    expert_policy: Mapping[str, Any] | None,
    personalization: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Install learning artifacts in shadow mode (no live influence)."""
    artifact = {
        "schema": SIGNAL_SCHEMA,
        "mode": "shadow",
        "cap": EXPERT_SIGNAL_CAP,
        "expert_policy_path": None if not expert_policy else expert_policy.get("path"),
        "expert_policy_fingerprint": None if not expert_policy else expert_policy.get("fingerprint"),
        "personalization_path": None if not personalization else personalization.get("path"),
        "personalization_fingerprint": None if not personalization else personalization.get("fingerprint"),
        "evaluation": dict(evaluation or {}),
        "playbook_replacement": False,
        "unverified_formations": False,
        "invented_adjustments": False,
        "model_primary_retained": True,
        "live_influence": False,
        "user_approved": False,
        "gate_passed": False,
    }
    save_expert_signal(db, artifact)
    if personalization is not None:
        shadow = dict(personalization, mode="shadow")
        save_personalization(db, shadow)
    return artifact


def evaluation_gates_passed(evaluation: Mapping[str, Any] | None) -> tuple[bool, list[str]]:
    if not evaluation:
        return False, ["missing_evaluation"]
    failures = []
    if not evaluation.get("match_holdout"):
        failures.append("match_holdout_required")
    if not evaluation.get("expert_holdout"):
        failures.append("expert_holdout_required")
    if evaluation.get("insufficient_data"):
        failures.append("insufficient_data")
    # Require non-degradation vs baseline when enough data exists.
    delta = evaluation.get("personalized_minus_expert")
    if delta is not None:
        try:
            if float(delta) < -0.02:
                failures.append("personalized_worse_than_expert")
        except (TypeError, ValueError):
            failures.append("invalid_personalized_delta")
    expert_delta = evaluation.get("expert_minus_baseline")
    if expert_delta is not None:
        try:
            if float(expert_delta) < -0.02:
                failures.append("expert_worse_than_baseline")
        except (TypeError, ValueError):
            failures.append("invalid_expert_delta")
    return (not failures), failures


def promote_expert_signal(db: Any, *, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Explicit user promotion after evaluation gates pass."""
    current = load_expert_signal(db) or {
        "schema": SIGNAL_SCHEMA,
        "mode": "shadow",
        "cap": EXPERT_SIGNAL_CAP,
    }
    report = dict(evaluation or current.get("evaluation") or {})
    ok, failures = evaluation_gates_passed(report)
    if not ok:
        return {
            "ok": False,
            "mode": current.get("mode", "shadow"),
            "gate_passed": False,
            "failures": failures,
            "live_influence": False,
            "user_approved": False,
        }
    promoted = dict(current)
    promoted.update({
        "mode": "bounded_active",
        "live_influence": True,
        "user_approved": True,
        "gate_passed": True,
        "evaluation": report,
        "cap": EXPERT_SIGNAL_CAP,
        "model_primary_retained": True,
        "playbook_replacement": False,
    })
    save_expert_signal(db, promoted)
    personal = load_personalization(db=db)
    if personal is not None:
        save_personalization(db, dict(personal, mode="bounded_active"))
    return {"ok": True, **promoted}


def rollback_expert_signal(db: Any) -> dict[str, Any]:
    current = load_expert_signal(db) or {"schema": SIGNAL_SCHEMA}
    rolled = dict(current)
    rolled.update({
        "mode": "shadow",
        "live_influence": False,
        "user_approved": False,
        "gate_passed": False,
    })
    save_expert_signal(db, rolled)
    personal = load_personalization(db=db)
    if personal is not None:
        save_personalization(db, dict(personal, mode="shadow"))
    return {"ok": True, **rolled}


def expert_learning_adjustment(
    play: str,
    sit: Any,
    *,
    db: Any = None,
    expert_policy: Mapping[str, Any] | None = None,
    personalization: Mapping[str, Any] | None = None,
    signal: Mapping[str, Any] | None = None,
    opponent_type: str = "cpu",
    book: Mapping[str, list[str]] | None = None,
    formation: str | None = None,
) -> dict[str, Any]:
    """Bounded score delta. Shadow mode reports influence without applying it."""
    active_signal = signal if signal is not None else load_expert_signal(db)
    mode = str((active_signal or {}).get("mode") or DEFAULT_MODE)
    cap = float((active_signal or {}).get("cap") or EXPERT_SIGNAL_CAP)
    policy = expert_policy
    if policy is None and active_signal and active_signal.get("expert_policy_path"):
        policy = load_expert_policy(path=active_signal.get("expert_policy_path"))
    personal = personalization
    if personal is None:
        personal = load_personalization(db=db)
        if personal is None and active_signal and active_signal.get("personalization_path"):
            personal = load_personalization(path=active_signal.get("personalization_path"))

    # Never invent formations outside the installed book.
    if book is not None and formation is not None:
        plays = book.get(formation) or []
        if play not in plays:
            return {
                "play": play,
                "delta": 0.0,
                "applied_delta": 0.0,
                "reasons": ["play not in installed playbook formation; expert signal withheld"],
                "withheld": ["unverified_or_missing_playbook_entry"],
                "cap": cap,
                "mode": mode,
                "live_influence": False,
                "shadow_delta": 0.0,
                "model_primary_retained": True,
            }

    blend = blend_for_play(
        play, sit=sit, expert_policy=policy, personalization=personal,
        opponent_type=opponent_type,
    )
    raw = float(blend.get("delta") or 0.0)
    bounded = max(-cap, min(cap, raw * cap))
    live = mode == "bounded_active" and bool((active_signal or {}).get("user_approved"))
    return {
        "play": play,
        "concept_id": blend.get("concept_id"),
        "delta": round(bounded if live else 0.0, 5),
        "applied_delta": round(bounded if live else 0.0, 5),
        "shadow_delta": round(bounded, 5),
        "reasons": list(blend.get("reasons") or []),
        "withheld": list(blend.get("withheld") or []),
        "weights": blend.get("weights") or {},
        "uncertainty": blend.get("uncertainty"),
        "conflict": blend.get("conflict"),
        "cap": cap,
        "mode": mode,
        "live_influence": live,
        "changes_decision": False,  # filled by caller comparing rankings when needed
        "model_primary_retained": True,
        "playbook_replacement": False,
        "invented_adjustments": False,
        "context_key": blend.get("context_key"),
    }
