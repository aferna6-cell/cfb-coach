"""In-game adaptive memory for the offensive coordinator.

Tracks recommendations, verified executions, outcomes and credible defensive
observations across a single game. Never leaks post-snap facts into the
current pre-snap decision. A suggested play is not an executed play.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping

META_GAME_MEMORY = "ml_offense_game_memory.v1:{game_id}"
VERSION = "offense_game_memory.v1"
MIN_LOOKS_FOR_TENDENCY = 3


def _key(game_id: str) -> str:
    return META_GAME_MEMORY.format(game_id=game_id or "unknown")


def empty_memory(game_id: str) -> dict[str, Any]:
    return {
        "schema": VERSION,
        "game_id": game_id,
        "updated_ts": None,
        "recommendations": [],
        "verified_executions": [],
        "verified_adjustments": [],
        "outcomes": [],
        "concept_usage": {},
        "formation_usage": {},
        "live_looks": [],
        "opponent_tendencies": {
            "n_looks": 0,
            "coverage_counts": {},
            "pressure_rate": 0.0,
            "modal_coverage": None,
            "confidence": "none",
            "note": "tendency is not current coverage",
        },
        "objectives": {"possession_objective": "neutral"},
        "model_versions": [],
    }


def load_game_memory(db: Any, game_id: str) -> dict[str, Any]:
    if db is None or not game_id:
        return empty_memory(game_id or "unknown")
    raw = db.get_meta(_key(game_id))
    if not raw:
        return empty_memory(game_id)
    try:
        data = json.loads(raw)
        if data.get("schema") == VERSION:
            return data
    except (TypeError, ValueError):
        pass
    return empty_memory(game_id)


def save_game_memory(db: Any, memory: Mapping[str, Any]) -> None:
    if db is None:
        return
    payload = dict(memory)
    payload["updated_ts"] = datetime.now(timezone.utc).isoformat()
    db.set_meta(_key(str(payload.get("game_id") or "unknown")), json.dumps(payload, sort_keys=True))


def record_recommendation(
    memory: dict[str, Any],
    *,
    snap_id: str | None,
    formation: str,
    play: str,
    adjustment_plan: Mapping[str, Any] | None,
    situation: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    legal_candidates: int | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Record what the model recommended — not what was executed."""
    from cfb_coach.madden.model.experimental_model import _play_concept

    mem = dict(memory)
    mem.setdefault("recommendations", []).append({
        "snap_id": snap_id,
        "formation": formation,
        "play": play,
        "concept": _play_concept(play),
        "adjustment_kind": (adjustment_plan or {}).get("kind", "none"),
        "adjustment_id": (adjustment_plan or {}).get("id"),
        "situation_bucket": (situation or {}).get("situation_bucket"),
        "rationale": rationale,
        "legal_candidates": legal_candidates,
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    concepts = Counter(mem.get("concept_usage") or {})
    concepts[_play_concept(play)] += 1
    mem["concept_usage"] = dict(concepts)
    forms = Counter(mem.get("formation_usage") or {})
    forms[formation] += 1
    mem["formation_usage"] = dict(forms)
    if model_version:
        versions = list(mem.get("model_versions") or [])
        if model_version not in versions:
            versions.append(model_version)
        mem["model_versions"] = versions[-8:]
    if situation and situation.get("possession_objective"):
        mem.setdefault("objectives", {})["possession_objective"] = situation["possession_objective"]
    return mem


def record_live_look(
    memory: dict[str, Any],
    *,
    coverage_class: str | None,
    coverage_source: str | None,
    snap_id: str | None = None,
) -> dict[str, Any]:
    """Accumulate only credible live looks into opponent tendencies."""
    mem = dict(memory)
    if coverage_source != "live" or not coverage_class:
        return mem
    looks = list(mem.get("live_looks") or [])
    looks.append({
        "snap_id": snap_id,
        "coverage_class": coverage_class,
        "source": "live",
        "ts": datetime.now(timezone.utc).isoformat(),
    })
    mem["live_looks"] = looks[-80:]
    counts: dict[str, int] = defaultdict(int)
    for row in mem["live_looks"]:
        counts[str(row["coverage_class"])] += 1
    n = sum(counts.values())
    pressure = counts.get("pressure", 0) / n if n else 0.0
    modal = max(counts, key=counts.get) if counts else None
    confidence = (
        "high" if n >= 8 else "medium" if n >= MIN_LOOKS_FOR_TENDENCY else "low" if n else "none"
    )
    mem["opponent_tendencies"] = {
        "n_looks": n,
        "coverage_counts": dict(counts),
        "pressure_rate": round(pressure, 4),
        "modal_coverage": modal,
        "confidence": confidence,
        "note": "tendency is not current coverage; last look is not proof of present look",
    }
    return mem


def record_verified_outcome(
    memory: dict[str, Any],
    *,
    snap_id: str,
    executed_formation: str | None,
    executed_play: str | None,
    executed_verification: str | None,
    adjustment_applied: Mapping[str, Any] | None = None,
    success: bool | None = None,
    result: str | None = None,
    turnover: bool = False,
    sack: bool = False,
) -> dict[str, Any]:
    """Attach verified execution / outcome facts after the snap ends."""
    mem = dict(memory)
    if executed_verification == "verified" and executed_formation and executed_play:
        mem.setdefault("verified_executions", []).append({
            "snap_id": snap_id,
            "formation": executed_formation,
            "play": executed_play,
        })
    if adjustment_applied and adjustment_applied.get("confirmed"):
        mem.setdefault("verified_adjustments", []).append({
            "snap_id": snap_id,
            "kind": adjustment_applied.get("kind"),
            "id": adjustment_applied.get("id"),
        })
    if success is not None or result is not None:
        mem.setdefault("outcomes", []).append({
            "snap_id": snap_id,
            "success": success,
            "result": result,
            "turnover": turnover,
            "sack": sack,
        })
    return mem


def decision_context(memory: Mapping[str, Any]) -> dict[str, Any]:
    """Safe pre-snap summary for the joint decision engine (no future leakage)."""
    tendencies = dict(memory.get("opponent_tendencies") or {})
    return {
        "opponent_tendencies": tendencies,
        "concept_usage": dict(memory.get("concept_usage") or {}),
        "formation_usage": dict(memory.get("formation_usage") or {}),
        "objectives": dict(memory.get("objectives") or {}),
        "n_recommendations": len(memory.get("recommendations") or []),
        "roster_verified": False,
    }


def update_from_decision(
    db: Any,
    *,
    game_id: str,
    snap_id: str | None,
    formation: str,
    play: str,
    adjustment_plan: Mapping[str, Any] | None,
    sit: Any,
    football_situation: Mapping[str, Any] | None = None,
    model_version: str | None = None,
    legal_candidates: int | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Persist recommendation + live look after a decision is sealed."""
    mem = load_game_memory(db, game_id)
    mem = record_recommendation(
        mem, snap_id=snap_id, formation=formation, play=play,
        adjustment_plan=adjustment_plan, situation=football_situation,
        model_version=model_version, legal_candidates=legal_candidates,
        rationale=rationale,
    )
    cov = getattr(sit, "coverage_hint", None)
    src = getattr(sit, "coverage_source", None)
    from cfb_coach.madden.playcaller import coverage_class

    cls = coverage_class(cov) if cov else None
    mem = record_live_look(mem, coverage_class=cls, coverage_source=src, snap_id=snap_id)
    save_game_memory(db, mem)
    return mem
