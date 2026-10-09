"""Revisable drive hypothesis for the offensive coordinator.

This is not a drive script. A hypothesis can nudge a near tie. It cannot
force a run to "establish the run," and it cannot override a clearly
stronger learned play.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from cfb_coach.madden.model.defensive_diagnosis import diagnose_defense
from cfb_coach.madden.model.football_knowledge import CONCEPTS, profile_for_play

STRATEGY_VERSION = "offense_strategy.v1"
SCHEMA = "madden.offense.strategy.v1"
META = "ml_offense_strategy.v1"
STRATEGY_CAP = 0.015
_UNDERNEATH = frozenset({
    "mesh", "stick", "slant_flat", "curl_flat", "spacing",
})
_UNDERNEATH_SHELLS = frozenset({"cover_2", "cover_3", "cover_4", "cover_6"})
_COMPLEMENT_BONUS = 0.008


def _key(session_id: str) -> str:
    return f"{META}:{session_id or 'unscoped'}"


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def drive_boundary(sit: Any, memory: Mapping[str, Any] | None = None) -> str:
    """New drive, continuing drive, or unknown. Unknown does not invent a script."""
    if sit is None or getattr(sit, "down", None) is None:
        return "unknown"
    down = _int(getattr(sit, "down", None))
    distance = _int(getattr(sit, "distance", None))
    if down is None:
        return "unknown"
    last = str((memory or {}).get("last_verified_outcome") or "").lower()
    if any(token in last.split() for token in ("td", "int", "punt", "turnover", "downs")):
        return "new_drive"
    if "fumble" in last:
        return "new_drive"
    if down == 1 and distance is not None and distance >= 10:
        return "new_drive"
    return "continuing"


def _concept_counts(memory: Mapping[str, Any] | None) -> tuple[dict[str, int], dict[str, int]]:
    block = (memory or {}).get("verified_concepts") or {}
    success = {str(k): int(v) for k, v in (block.get("success") or {}).items()}
    failure = {str(k): int(v) for k, v in (block.get("failure") or {}).items()}
    return success, failure


def _objective(sit: Any, diagnosis: Mapping[str, Any], memory: Mapping[str, Any] | None) -> str:
    down = _int(getattr(sit, "down", None)) if sit is not None else None
    distance = _int(getattr(sit, "distance", None)) if sit is not None else None
    quarter = _int(getattr(sit, "quarter", None)) if sit is not None else None
    if quarter is None and sit is not None:
        quarter = _int((getattr(sit, "extras", None) or {}).get("quarter"))
    score_us = getattr(sit, "score_us", None) if sit is not None else None
    score_them = getattr(sit, "score_them", None) if sit is not None else None
    two_minute = bool(getattr(sit, "two_minute", False)) if sit is not None else False
    goal_line = bool(getattr(sit, "goal_line", False)) if sit is not None else False
    if goal_line or (down in (3, 4) and distance is not None and distance <= 2):
        return "convert_critical_first_down"
    if down in (3, 4) and distance is not None and distance >= 7:
        return "convert_critical_first_down"
    trailing = (
        score_us is not None and score_them is not None and int(score_us) < int(score_them)
    )
    leading = (
        score_us is not None and score_them is not None and int(score_us) > int(score_them)
    )
    if two_minute and trailing and quarter == 2:
        return "score_before_halftime"
    if two_minute and trailing:
        return "create_explosive"
    if two_minute and leading:
        return "protect_late_lead"
    if quarter is not None and quarter >= 4 and leading:
        return "protect_late_lead"
    if quarter is not None and quarter >= 4 and trailing and not two_minute:
        return "conserve_clock_while_trailing"
    observed = diagnosis.get("observed") or {}
    if diagnosis.get("state") == "observed" and observed.get("pressure"):
        return "handle_pressure"
    if int((memory or {}).get("pressure_observations") or 0) >= 3:
        return "handle_pressure"
    inferred = (diagnosis.get("inferred") or {}).get("shell")
    if inferred in _UNDERNEATH_SHELLS:
        return "exploit_recurring_alignment"
    return "sustain_possession"


def current_strategy(
    sit: Any = None,
    memory: Mapping[str, Any] | None = None,
    diagnosis: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Hypothesis for this snap. A missing drive boundary keeps the prior hypothesis."""
    if diagnosis is None:
        diagnosis = diagnose_defense(sit, memory)
    boundary = drive_boundary(sit, memory)
    success, failure = _concept_counts(memory)
    mesh_failed = failure.get("mesh", 0) >= 3 and success.get("mesh", 0) == 0
    inferred = (diagnosis.get("inferred") or {}).get("shell")
    observed = diagnosis.get("observed") or {}
    underneath = (
        not mesh_failed
        and (
            inferred in _UNDERNEATH_SHELLS
            or (
                diagnosis.get("state") == "observed"
                and observed.get("shell") in _UNDERNEATH_SHELLS
            )
        )
    )
    if mesh_failed:
        hypothesis_id = "revised_away_from_underneath"
        hypothesis = (
            "Verified results contradict an underneath script. "
            "Mesh has failed without a verified success."
        )
        revised = True
    elif underneath:
        hypothesis_id = "underneath_space"
        hypothesis = (
            "Opponent frequently exposes underneath space while protecting deep zones."
        )
        revised = False
    elif _objective(sit, diagnosis, memory) == "handle_pressure":
        hypothesis_id = "pressure_answers"
        hypothesis = (
            "Pressure has shown up often enough to respect it. "
            "It is not automatically this snap's coverage."
        )
        revised = False
    else:
        hypothesis_id = "none"
        hypothesis = "No defensive tendency is strong enough to set a concept script."
        revised = False

    objective = _objective(sit, diagnosis, memory)
    kept_previous = False
    if boundary == "unknown" and previous and previous.get("hypothesis"):
        hypothesis_id = str(previous.get("hypothesis_id") or hypothesis_id)
        hypothesis = str(previous.get("hypothesis") or hypothesis)
        objective = str(previous.get("objective") or objective)
        kept_previous = True
        revised = bool(previous.get("revised"))

    confidence = 0.0
    if hypothesis_id == "underneath_space":
        confidence = float((diagnosis.get("inferred") or {}).get("confidence") or 0.0)
        if diagnosis.get("state") == "observed" and observed.get("shell") in _UNDERNEATH_SHELLS:
            confidence = max(confidence, 0.55)
    elif hypothesis_id == "revised_away_from_underneath":
        confidence = 0.7
    gaps = [
        "route_diagrams_unknown",
        "personnel_attributes_unknown",
        "run_fit_unknown",
    ]
    if "shell" in (diagnosis.get("unknown") or []):
        gaps.append("current_shell_unknown")
    if boundary == "unknown":
        gaps.append("drive_boundary_unknown")
    return {
        "version": STRATEGY_VERSION,
        "schema": SCHEMA,
        "objective": objective,
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "confidence": round(confidence, 3),
        "drive_boundary": boundary,
        "kept_previous_hypothesis": kept_previous,
        "revised": revised,
        "not_a_script": True,
        "concept_successes": success,
        "concept_failures": failure,
        "recent_verified_concept": (memory or {}).get("recent_verified_concept"),
        "information_gaps": gaps,
        "note": (
            "Hypothesis only. Complementary plays are preferred only by a "
            "small capped delta, and a run is not forced to establish the run."
        ),
    }


def strategy_adjustment(
    play: str,
    strategy: Mapping[str, Any] | None,
    *,
    knowledge_delta: float = 0.0,
) -> dict[str, Any]:
    """Small preference. Zero when knowledge already scored the same idea."""
    profile = profile_for_play(play)
    concept_id = profile.get("concept_id")
    reasons: list[str] = []
    delta = 0.0
    if not strategy or not concept_id:
        return _adjustment(play, concept_id, 0.0, ["no strategy hypothesis"], [])
    failures = int((strategy.get("concept_failures") or {}).get(concept_id) or 0)
    successes = int((strategy.get("concept_successes") or {}).get(concept_id) or 0)
    if failures >= 3 and successes == 0:
        return _adjustment(
            play, concept_id, 0.0,
            [f"verified results revised {concept_id} out of the hypothesis"],
            ["contradicted_concept"],
        )
    if float(knowledge_delta) > 0:
        reasons.append(
            "knowledge already scored this concept; strategy does not add the same prior"
        )
    elif (
        strategy.get("hypothesis_id") == "underneath_space"
        and concept_id in _UNDERNEATH
    ):
        delta = STRATEGY_CAP
        reasons.append("underneath hypothesis favors this concept family")
    principle = CONCEPTS.get(concept_id) or {}
    recent = strategy.get("recent_verified_concept")
    if (
        recent
        and recent in (principle.get("complements") or [])
        and delta + _COMPLEMENT_BONUS <= STRATEGY_CAP
        and float(knowledge_delta) <= 0
        and delta == 0
    ):
        delta = _COMPLEMENT_BONUS
        reasons.append(
            f"{concept_id} complements verified {recent}; preference only, not a script"
        )
    if concept_id in ("inside_zone", "outside_zone", "power", "dive", "qb_run"):
        reasons.append("a run is not added merely to establish the run")
    if strategy.get("objective") == "conserve_clock_while_trailing":
        reasons.append(
            "trailing late; do not kill clock with a forced run, and do not add a second pass bonus"
        )
    if not reasons:
        reasons.append("strategy does not move this play")
    return _adjustment(play, concept_id, min(STRATEGY_CAP, delta), reasons, [])


def _adjustment(
    play: str,
    concept_id: str | None,
    delta: float,
    reasons: list[str],
    withheld: list[str],
) -> dict[str, Any]:
    return {
        "play": play,
        "concept_id": concept_id,
        "delta": round(min(STRATEGY_CAP, max(0.0, delta)), 5),
        "reasons": reasons,
        "withheld": withheld,
        "cap": STRATEGY_CAP,
        "not_a_script": True,
    }


def load_strategy(db: Any, session_id: str) -> dict[str, Any] | None:
    if db is None:
        return None
    raw = db.get_meta(_key(session_id))
    try:
        payload = json.loads(raw) if raw else None
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        return None
    return payload


def save_strategy(db: Any, session_id: str, state: Mapping[str, Any]) -> None:
    """Store the hypothesis only. Snap history is not rewritten."""
    if db is None:
        return
    payload = {
        "schema": SCHEMA,
        "session_id": session_id,
        "objective": state.get("objective"),
        "hypothesis_id": state.get("hypothesis_id"),
        "hypothesis": state.get("hypothesis"),
        "confidence": state.get("confidence"),
        "revised": bool(state.get("revised")),
        "drive_boundary": state.get("drive_boundary"),
        "not_a_script": True,
    }
    db.set_meta(_key(session_id), json.dumps(payload, sort_keys=True))


def strategy_report(db: Any = None, opponent_id: str = "cpu") -> dict[str, Any]:
    """Read-only game plan. Does not install formations or rewrite history."""
    staged = None
    profile = None
    if db is not None:
        try:
            from cfb_coach.madden.model.offense_designer import staged_design

            staged = staged_design(db)
        except Exception:  # noqa: BLE001
            staged = None
        try:
            from cfb_coach.madden.model.offense_portfolio import opponent_defense_profile

            profile = opponent_defense_profile(db, opponent_id)
        except Exception:  # noqa: BLE001
            profile = None
    diagnosis = diagnose_defense(None, None)
    plan = current_strategy(None, None, diagnosis, previous=None)
    selected: list[dict[str, Any]] = []
    coverage = None
    football_plan = None
    if staged:
        book = staged.get("book") or {}
        rationales = {
            row.get("formation"): row for row in (staged.get("formation_rationales") or [])
        }
        for form, plays in (book.get("formations") or {}).items():
            why = (rationales.get(form) or {}).get("why")
            selected.append({
                "formation": form,
                "plays": len(plays or []),
                "why": why,
            })
        coverage = staged.get("situation_coverage")
        football_plan = staged.get("football_plan")
    gaps = list(plan["information_gaps"])
    if not staged:
        gaps.append("no_staged_playbook")
    if profile is None or profile.get("state") not in ("inferred", "verified"):
        gaps.append("opponent_tendency_unconfirmed")
    return {
        "version": STRATEGY_VERSION,
        "opponent_id": opponent_id,
        "read_only": True,
        "history_modified": False,
        "installs_formations": False,
        "game_plan": plan,
        "selected_formations": selected,
        "situation_coverage": coverage,
        "football_plan": football_plan,
        "confidence": plan["confidence"],
        "information_gaps": gaps,
        "staged_proposal_id": (staged or {}).get("proposal_id"),
        "opponent_tendency": profile,
        "note": (
            "Diagnostic only. Fifteen formations is the maximum, not a quota. "
            "Nothing is installed by this command."
        ),
    }
