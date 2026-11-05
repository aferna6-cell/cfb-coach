"""Model-discretionary Madden offensive formation pool, with no timer/round-robin.

This changes only which *already installed* formations the coordinator treats
as strategically active. It never mutates the applied/physical Madden book;
an outside challenger remains legal and can earn entry whenever model evidence
justifies it. A new physical formation requires the existing staged design
and explicit in-game installation attestation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

POLICY = "model_discretionary_formation_pool.v1"
ENTER_GAP = 0.052
STAY_GAP = 0.082
OUTSIDE_PENALTY = 0.012  # soft preference only; NEVER excludes a legal play


def _score(row: Mapping[str, Any]) -> float:
    return float(row.get("selection_score", row.get("probability", 0.0)) or 0.0)


def _state_key(opponent_id: str, session_id: str | None) -> str:
    # Session-scoped; state never contaminates the next game or another opponent.
    return "ml_offense_formation_pool.v1:" + hashlib.sha256(
        f"{opponent_id}|{session_id or 'unscoped'}".encode("utf-8")
    ).hexdigest()[:18]


def _read_state(db: Any, key: str) -> dict[str, Any]:
    try:
        value = db.get_meta(key) if db is not None else None
        raw = json.loads(value) if value else {}
        return raw if isinstance(raw, dict) else {}
    except (ValueError, TypeError, AttributeError):
        return {}


def choose_active_formations(
    ranked: Sequence[Mapping[str, Any]], *,
    previous: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Decide active formations from current modeled upside, not snap count.

    Entry/stay thresholds form an evidence-based hysteresis band. A formation
    can remain as long as competitive and can enter at the very next call.
    No fixed maximum active count, no forced substitution, no rotation period.
    """
    best_by_form: dict[str, float] = {}
    for item in ranked:
        form = str(item.get("formation") or "")
        if form:
            best_by_form[form] = max(best_by_form.get(form, -1.0), _score(item))
    if not best_by_form:
        return [], {"reason": "no legal formation", "entered": [], "exited": []}
    ceiling = max(best_by_form.values())
    prior = set(str(x) for x in previous) & set(best_by_form)
    active = {
        f for f, value in best_by_form.items()
        if ceiling - value <= (STAY_GAP if f in prior else ENTER_GAP)
    }
    if not active:
        active = {max(best_by_form, key=best_by_form.get)}
    # No fixed quota. If all formations genuinely have comparable predictive
    # scores, all of them may remain active.
    ordered = sorted(active, key=lambda f: (-best_by_form[f], f))
    return ordered, {
        "reason": "formation model scores crossed relative entry/stay bands",
        "entered": sorted(active - prior),
        "exited": sorted(prior - active),
        "best_score": round(ceiling, 6),
        "installed_candidates": len(best_by_form),
        "active_count": len(ordered),
        "scheduled_rotation": False,
    }


def apply_formation_rotation(
    ranked: Sequence[Mapping[str, Any]], *,
    db: Any = None, opponent_id: str = "",
    session_id: str | None = None,
    persist: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Soft-pool ranking; entire installed inventory stays eligible.

    State is saved only if the actual modeled active set changes. If DB meta is
    unavailable, calls remain functional with stateless model discretion.
    """
    if not ranked:
        return [], {"policy": POLICY, "active": [], "eligible_count": 0}
    key = _state_key(opponent_id, session_id)
    previous = _read_state(db, key) if session_id else {}
    installed = sorted({str(r["formation"]) for r in ranked})
    prior = previous.get("active") or []
    if sorted(previous.get("installed") or []) != installed:
        prior = []  # physically confirmed playbook changed; re-score freely

    active, reason = choose_active_formations(ranked, previous=prior)
    chosen = set(active)
    rescored = []
    for item in ranked:
        r = dict(item)
        r["formation_pool_state"] = "active" if r["formation"] in chosen else "challenger"
        r["formation_pool_penalty"] = 0.0 if r["formation"] in chosen else OUTSIDE_PENALTY
        # A challenger can still win selection, triggering next-snap re-ranking.
        r["selection_score"] = round(_score(r) - r["formation_pool_penalty"], 7)
        rescored.append(r)
    rescored.sort(key=lambda r: (-float(r["selection_score"]), -float(r["probability"]),
                                 r["formation"], r["play"]))

    if persist and db is not None and session_id and (
        sorted(prior) != sorted(active) or previous.get("installed") != installed
    ):
        try:
            db.set_meta(key, json.dumps({
                "policy": POLICY, "installed": installed, "active": active,
                "basis": "situational_model_success", "no_rotation_schedule": True,
            }, sort_keys=True))
        except Exception:  # noqa: BLE001 - rotation metadata cannot block live play
            reason["state_write_failed"] = True

    return rescored, {
        "policy": POLICY,
        "active": active,
        "challengers": sorted(set(installed) - chosen),
        "eligible_count": len(ranked),
        **reason,
        "soft_pool_only": True,
        "no_physical_book_change": True,
    }
