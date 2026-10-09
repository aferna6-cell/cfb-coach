"""In-game offensive memory. Recommendations and results stay separate.

A suggested play is not an executed play. A suggested adjustment is not an
applied adjustment. Context for snap N never includes snap N's outcome, and
it never treats an earlier coverage observation as the current coverage.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

SCHEMA = "madden.offense.game_memory.v1"
META = "ml_offense_game_memory.v1"


def _key(session_id: str) -> str:
    return f"{META}:{session_id or 'unscoped'}"


def _load(db: Any, session_id: str) -> dict[str, Any]:
    raw = db.get_meta(_key(session_id)) if db is not None else None
    try:
        payload = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        payload = {}
    if payload.get("schema") != SCHEMA:
        payload = {"schema": SCHEMA, "session_id": session_id, "events": []}
    payload.setdefault("events", [])
    return payload


def _save(db: Any, session_id: str, payload: Mapping[str, Any]) -> None:
    db.set_meta(_key(session_id), json.dumps(dict(payload), sort_keys=True))


def remember_recommendation(
    db: Any,
    *,
    session_id: str,
    snap_id: str | None,
    snap_seq: int | None,
    formation: str,
    play: str,
    adjustment_kind: str = "none",
    adjustment_id: str | None = None,
    presnap: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Store what was shown. This does not label the snap as executed."""
    payload = _load(db, session_id)
    event = {
        "snap_id": snap_id,
        "snap_seq": snap_seq,
        "recommended_formation": formation,
        "recommended_play": play,
        "recommended_adjustment_kind": adjustment_kind or "none",
        "recommended_adjustment_id": adjustment_id,
        "presnap": dict(presnap or {}),
        "executed_formation": None,
        "executed_play": None,
        "adjustment_applied": None,
        "no_adjustment_confirmed": None,
        "outcome": None,
        "observed_defense": None,
        "verification": "recommendation_only",
    }
    events = [
        row for row in payload["events"]
        if not (snap_id and row.get("snap_id") == snap_id)
    ]
    events.append(event)
    payload["events"] = events[-80:]
    _save(db, session_id, payload)
    return event


def record_closed_snap(
    db: Any,
    *,
    session_id: str,
    snap_id: str | None,
    snap_seq: int | None = None,
    executed_formation: str | None = None,
    executed_play: str | None = None,
    adjustment_applied: bool | None = None,
    no_adjustment_confirmed: bool | None = None,
    observed_defense: str | None = None,
    outcome: str | None = None,
    verified: bool = False,
) -> dict[str, Any]:
    """Attach a closed snap to its recommendation without merging the two facts."""
    payload = _load(db, session_id)
    match = None
    for row in payload["events"]:
        if snap_id and row.get("snap_id") == snap_id:
            match = row
            break
    if match is None:
        match = {
            "snap_id": snap_id,
            "snap_seq": snap_seq,
            "recommended_formation": None,
            "recommended_play": None,
            "recommended_adjustment_kind": None,
            "recommended_adjustment_id": None,
            "presnap": {},
            "verification": "outcome_without_recommendation",
        }
        payload["events"].append(match)
    match["executed_formation"] = executed_formation
    match["executed_play"] = executed_play
    match["adjustment_applied"] = adjustment_applied
    match["no_adjustment_confirmed"] = no_adjustment_confirmed
    match["observed_defense"] = observed_defense
    match["outcome"] = outcome
    match["verification"] = "verified_execution" if verified else "unverified_result"
    _save(db, session_id, payload)
    return match


def pre_snap_context(
    db: Any,
    *,
    session_id: str,
    snap_seq: int | None,
    snap_id: str | None = None,
) -> dict[str, Any]:
    """Tendencies available before this snap. Later outcomes are invisible."""
    payload = _load(db, session_id) if db is not None else {"events": []}
    prior: list[dict[str, Any]] = []
    for row in payload.get("events") or []:
        if snap_id and row.get("snap_id") == snap_id:
            continue
        seq = row.get("snap_seq")
        if snap_seq is not None and seq is not None and int(seq) >= int(snap_seq):
            continue
        prior.append(row)
    from cfb_coach.madden.playcaller import coverage_class

    counts: dict[str, int] = {}
    for row in prior:
        # Only a defense actually seen on an already closed snap.
        kind = coverage_class(row.get("observed_defense"))
        if kind and row.get("outcome") is not None:
            counts[kind] = counts.get(kind, 0) + 1
    sample = sum(counts.values())
    inferred = max(counts, key=lambda k: (counts[k], k)) if counts else None
    state = "inferred" if sample >= 3 and inferred else "unknown"
    verified = [row for row in prior if row.get("verification") == "verified_execution"]
    failures: dict[str, int] = {}
    successes: dict[str, int] = {}
    for row in verified:
        play = row.get("executed_play")
        if not play:
            continue
        text = str(row.get("outcome") or "").lower()
        bad = any(tok in text for tok in ("int", "fumble", "sack", "incomplete"))
        if bad:
            failures[play] = failures.get(play, 0) + 1
        else:
            successes[play] = successes.get(play, 0) + 1
    discouraged = [
        play for play, n in failures.items()
        if n >= 4 and successes.get(play, 0) == 0
    ]
    observed_labels = [
        str(row.get("observed_defense"))
        for row in prior
        if row.get("outcome") is not None and row.get("observed_defense")
    ]
    concept_success: dict[str, int] = {}
    concept_failure: dict[str, int] = {}
    recent_verified_concept = None
    try:
        from cfb_coach.madden.model.football_knowledge import profile_for_play
    except Exception:  # noqa: BLE001
        profile_for_play = None  # type: ignore[assignment]
    last_verified_outcome = None
    for row in verified:
        last_verified_outcome = row.get("outcome")
        if profile_for_play is None:
            continue
        concept_id = profile_for_play(row.get("executed_play")).get("concept_id")
        if not concept_id:
            continue
        text = str(row.get("outcome") or "").lower()
        bad = any(tok in text for tok in ("int", "fumble", "sack", "incomplete"))
        bucket = concept_failure if bad else concept_success
        bucket[concept_id] = bucket.get(concept_id, 0) + 1
        if not bad:
            recent_verified_concept = concept_id
    try:
        from cfb_coach.madden.model.opponent_learning import records_from_memory_events

        verified_snap_records = records_from_memory_events(verified)
    except Exception:  # noqa: BLE001 — memory still returns without the learner
        verified_snap_records = []
    return {
        "schema": SCHEMA,
        "events_before_snap": len(prior),
        "sample_size": sample,
        "confidence": round(min(1.0, sample / 6.0), 3) if state == "inferred" else 0.0,
        "distribution": counts,
        "inferred_look": inferred if state == "inferred" else None,
        "state": state,
        "pressure_observations": counts.get("pressure", 0),
        "note": "historical tendency from closed snaps; not the current coverage",
        "recommendations": sum(1 for row in prior if row.get("recommended_play")),
        "verified_executions": len(verified),
        "discouraged_plays": discouraged,
        "discouraged_rule": "four verified failures and zero verified successes; one snap is not enough",
        "observed_labels": observed_labels,
        "verified_concepts": {
            "success": concept_success,
            "failure": concept_failure,
        },
        "recent_verified_concept": recent_verified_concept,
        "last_verified_outcome": last_verified_outcome,
        "recent_recommendations": _recent_recommendations(prior),
        "verified_snap_records": verified_snap_records,
    }


def _recent_recommendations(prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shown calls before this snap. A recommendation is not an execution."""
    shown: list[dict[str, Any]] = []
    for row in reversed(prior):
        play = row.get("recommended_play")
        if not play:
            continue
        shown.append({
            "formation": row.get("recommended_formation"),
            "play": play,
            "adjustment_kind": row.get("recommended_adjustment_kind") or "none",
            "status": "recommended_not_verified_execution",
        })
        if len(shown) >= 20:
            break
    return shown
