"""Confidence-aware personalization over expert foundation + personal evidence.

Weights are estimated by context and concept from effective sample size,
uncertainty, contextual similarity, outcome reliability, and source quality.
There is no fixed game count or fixed percentage transition schedule.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.experimental_model import _dd_bucket, _play_concept
from cfb_coach.madden.model.football_knowledge import profile_for_play
from cfb_coach.madden.model.learning_sources import (
    ensure_store,
    fingerprint_payload,
    list_evidence,
    personal_rows_from_verified_snaps,
    retain_evidence,
    utc_now,
)
from cfb_coach.madden.model.expert_policy import (
    PRIOR_STRENGTH,
    load_expert_policy,
    score_play_under_expert_policy,
)

PERSONALIZATION_SCHEMA = "madden.personalization.v1"
PERSONALIZATION_VERSION = "personalization.v1"
META_PERSONALIZATION = "ml_offense_personalization.v1"
# Soft threshold for when personal evidence begins to dominate a context cell.
# Not a fixed game count — depends on effective sample size and reliability.
PERSONAL_PRIOR = 6.0
CONFLICT_PENALTY = 0.45


def ingest_personal_from_snaps(
    learning_store: str | Path,
    snaps: Sequence[Mapping[str, Any]],
    *,
    playbook: Mapping[str, Sequence[str]] | None = None,
    roster: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = personal_rows_from_verified_snaps(snaps, playbook=playbook, roster=roster)
    saved = 0
    rejected = 0
    for row in rows:
        result = retain_evidence(learning_store, row)
        if result.get("ok"):
            saved += 1
        else:
            rejected += 1
    return {
        "ok": True,
        "saved": saved,
        "rejected": rejected,
        "candidates": len(rows),
        "unexecuted_recommendations_excluded": True,
        "category": "personal_evidence",
    }


def _success_value(outcome: Mapping[str, Any] | None) -> float | None:
    if not outcome:
        return None
    success = outcome.get("success")
    if isinstance(success, bool):
        return 1.0 if success else 0.0
    if str(success).lower() in ("true", "1", "success"):
        return 1.0
    if str(success).lower() in ("false", "0", "failure"):
        return 0.0
    return None


def _context_key(row: Mapping[str, Any]) -> str:
    sit = row.get("situation") or {}
    action = row.get("observed_action") or {}
    play = action.get("play")
    concept = profile_for_play(play).get("concept_id") if play else None
    concept = concept or (_play_concept(play) if play else "unknown")
    return "|".join([
        _dd_bucket(sit.get("down"), sit.get("distance")),
        str(sit.get("opponent_category") or sit.get("opponent_type") or "unknown"),
        str(concept or "unknown"),
    ])


def fit_personalization(
    learning_store: str | Path,
    *,
    expert_policy: Mapping[str, Any] | None = None,
    artifact_dir: str | Path | None = None,
    personal_rows: Sequence[Mapping[str, Any]] | None = None,
    expert_rows: Sequence[Mapping[str, Any]] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Estimate per-context personalization weights without a fixed transition schedule."""
    personal = list(personal_rows) if personal_rows is not None else list_evidence(learning_store, category="personal_evidence")
    expert = list(expert_rows) if expert_rows is not None else list_evidence(learning_store, category="expert_evidence")
    policy = expert_policy or load_expert_policy(store=learning_store) or {}

    cell_n: dict[str, float] = defaultdict(float)
    cell_success: dict[str, float] = defaultdict(float)
    cell_games: dict[str, set[str]] = defaultdict(set)
    for row in personal:
        key = _context_key(row)
        value = _success_value(row.get("verified_outcome"))
        # Unknown outcomes are not failures and are excluded from personal rates.
        if value is not None:
            cell_n[key] += 1.0
            cell_success[key] += value
        if row.get("match_id"):
            cell_games[key].add(str(row["match_id"]))

    expert_pref: dict[str, float] = defaultdict(float)
    expert_n: dict[str, float] = defaultdict(float)
    for row in expert:
        sit = row.get("situation") or {}
        action = row.get("observed_action") or {}
        play = action.get("play")
        concept = action.get("concept_family") or (
            profile_for_play(play).get("concept_id") if play else None
        ) or (_play_concept(play) if play else None)
        if not concept:
            continue
        key = "|".join([
            _dd_bucket(sit.get("down"), sit.get("distance")),
            str(sit.get("opponent_type") or "unknown"),
            str(concept),
        ])
        weight = float(row.get("confidence") or 0.5)
        expert_n[key] += weight
        # Demonstration mass — not an outcome claim.
        expert_pref[key] += weight

    contexts = {}
    evolution = []
    for key, n in sorted(cell_n.items()):
        personal_rate = None
        if n > 0 and key in cell_success:
            personal_rate = cell_success[key] / n
        expert_mass = expert_n.get(key, 0.0)
        # Effective sample size with reliability discount when outcomes are sparse.
        reliability = 0.0 if personal_rate is None else min(1.0, n / (n + 2.0))
        personal_ess = n * reliability
        expert_ess = expert_mass
        # Similarity fallback: if this exact cell is empty, rely on expert/global.
        if personal_ess <= 0 and expert_ess <= 0:
            personal_weight = 0.0
            expert_weight = 0.0
            general_weight = 1.0
            conflict = False
            uncertainty = 1.0
        else:
            personal_weight = personal_ess / (personal_ess + PERSONAL_PRIOR + expert_ess * 0.5)
            expert_weight = (1.0 - personal_weight) * (
                expert_ess / (expert_ess + PRIOR_STRENGTH)
            )
            general_weight = max(0.0, 1.0 - personal_weight - expert_weight)
            conflict = False
            if personal_rate is not None and expert_mass > 0 and personal_rate < 0.4:
                # Personal outcomes contradict using this concept often.
                conflict = True
                personal_weight *= (1.0 - CONFLICT_PENALTY)
                expert_weight *= (1.0 - CONFLICT_PENALTY * 0.5)
                # Renormalize remaining mass toward uncertainty/general.
                total = personal_weight + expert_weight
                if total < 1.0:
                    general_weight = 1.0 - total
            uncertainty = round(
                1.0 / (1.0 + personal_ess + 0.5 * expert_ess),
                5,
            )
        contexts[key] = {
            "personal_n": round(n, 3),
            "personal_ess": round(personal_ess, 3),
            "personal_success_rate": None if personal_rate is None else round(personal_rate, 5),
            "personal_games": len(cell_games.get(key) or []),
            "expert_ess": round(expert_ess, 3),
            "personal_weight": round(personal_weight, 5),
            "expert_weight": round(expert_weight, 5),
            "general_weight": round(general_weight, 5),
            "uncertainty": uncertainty,
            "conflict": conflict,
            "opponent_specific": "cpu" in key or "human" in key,
        }
        evolution.append({
            "context": key,
            "personal_n": round(n, 3),
            "personal_weight": round(personal_weight, 5),
            "expert_weight": round(expert_weight, 5),
        })

    artifact = {
        "schema": PERSONALIZATION_SCHEMA,
        "version": PERSONALIZATION_VERSION,
        "fitted_at": utc_now(),
        "personal_rows": len(personal),
        "expert_rows": len(expert),
        "contexts": contexts,
        "evolution": evolution,
        "fixed_game_threshold": False,
        "fixed_percentage_schedule": False,
        "opponent_adaptation_distinct": True,
        "expert_policy_fingerprint": (policy or {}).get("fingerprint"),
        "mode": "shadow",
        "note": (
            "Limited personal data increases expert reliance; strong personal "
            "evidence increases personal influence; unfamiliar contexts fall "
            "back to expert/general knowledge; conflicts raise uncertainty."
        ),
    }
    artifact["fingerprint"] = fingerprint_payload({
        k: artifact[k] for k in ("schema", "version", "contexts", "personal_rows", "expert_rows")
    })
    if persist:
        paths = ensure_store(learning_store)
        out_dir = Path(artifact_dir) if artifact_dir else paths["artifacts"] / "personalization"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "personalization.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        artifact["path"] = str(path)
    return artifact


def load_personalization(
    path: str | Path | None = None,
    store: str | Path | None = None,
    db: Any = None,
) -> dict[str, Any] | None:
    if path is not None:
        candidate = Path(path)
    elif store is not None:
        candidate = ensure_store(store)["artifacts"] / "personalization" / "personalization.json"
    else:
        candidate = None
    if candidate is not None and candidate.is_file():
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, dict):
            return payload
    if db is not None:
        try:
            raw = db.get_meta(META_PERSONALIZATION)
        except Exception:  # noqa: BLE001
            raw = None
        if raw:
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None
            return payload if isinstance(payload, dict) else None
    return None


def save_personalization(db: Any, artifact: Mapping[str, Any]) -> None:
    if db is None:
        return
    db.set_meta(META_PERSONALIZATION, json.dumps(dict(artifact), sort_keys=True))


def blend_for_play(
    play: str,
    *,
    sit: Any,
    expert_policy: Mapping[str, Any] | None,
    personalization: Mapping[str, Any] | None,
    opponent_type: str = "cpu",
    competitive_mode: str = "unknown",
) -> dict[str, Any]:
    """Combine expert preference with personal context weights into one bounded score in [-1, 1]."""
    down = getattr(sit, "down", None) if sit is not None and not isinstance(sit, Mapping) else (
        (sit or {}).get("down") if isinstance(sit, Mapping) else None
    )
    distance = getattr(sit, "distance", None) if sit is not None and not isinstance(sit, Mapping) else (
        (sit or {}).get("distance") if isinstance(sit, Mapping) else None
    )
    expert = score_play_under_expert_policy(
        expert_policy, play,
        down=down, distance=distance, opponent_type=opponent_type,
        competitive_mode=competitive_mode,
    )
    concept = expert.get("concept_id") or _play_concept(play)
    key = "|".join([
        _dd_bucket(down, distance),
        str(opponent_type or "unknown"),
        str(concept or "unknown"),
    ])
    cell = ((personalization or {}).get("contexts") or {}).get(key) or {}
    if not cell:
        # Fall back to same down/distance concept with unknown opponent, then general.
        alt = "|".join([_dd_bucket(down, distance), "unknown", str(concept or "unknown")])
        cell = ((personalization or {}).get("contexts") or {}).get(alt) or {
            "personal_weight": 0.0,
            "expert_weight": 0.65 if expert_policy else 0.0,
            "general_weight": 0.35 if expert_policy else 1.0,
            "uncertainty": 1.0,
            "conflict": False,
            "personal_success_rate": None,
            "personal_ess": 0.0,
            "expert_ess": float(expert.get("effective_n") or 0.0),
        }
    personal_rate = cell.get("personal_success_rate")
    # Map personal success to [-1, 1] centered at 0.5.
    personal_score = 0.0 if personal_rate is None else (2.0 * float(personal_rate) - 1.0)
    expert_score = float(expert.get("delta") or 0.0)
    pw = float(cell.get("personal_weight") or 0.0)
    ew = float(cell.get("expert_weight") or 0.0)
    gw = float(cell.get("general_weight") or 0.0)
    blended = pw * personal_score + ew * expert_score + gw * 0.0
    if cell.get("conflict"):
        blended *= (1.0 - CONFLICT_PENALTY)
    uncertainty = float(cell.get("uncertainty") or 1.0)
    blended *= (1.0 - 0.5 * uncertainty)
    reasons = list(expert.get("reasons") or [])
    reasons.append(
        f"personalization weights p={pw:.2f} e={ew:.2f} g={gw:.2f} "
        f"uncertainty={uncertainty:.2f}"
    )
    if cell.get("conflict"):
        reasons.append("expert and personal evidence conflict; confidence reduced")
    if personal_rate is not None:
        reasons.append(f"personal success rate {personal_rate:.3f} (n_ess={cell.get('personal_ess')})")
    return {
        "delta": round(max(-1.0, min(1.0, blended)), 5),
        "reasons": reasons,
        "withheld": [],
        "concept_id": concept,
        "context_key": key,
        "weights": {
            "personal": round(pw, 5),
            "expert": round(ew, 5),
            "general": round(gw, 5),
        },
        "uncertainty": uncertainty,
        "conflict": bool(cell.get("conflict")),
        "personal_success_rate": personal_rate,
        "expert_component": expert_score,
        "personal_component": personal_score,
        "opponent_adaptation_distinct": True,
        "fixed_transition_schedule": False,
    }


def chronological_personal_split(
    snaps: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """For each snap, expose only previous games/snaps as eligible personal history."""
    ordered = sorted(
        snaps,
        key=lambda row: (
            str(row.get("game_id") or row.get("session_id") or ""),
            int(row.get("snap_seq") or row.get("id") or 0),
        ),
    )
    history: list[Mapping[str, Any]] = []
    views = []
    for row in ordered:
        views.append({
            "snap_id": row.get("snap_id") or row.get("ml_snap_id"),
            "eligible_history": list(history),
            "eligible_n": len(history),
        })
        history.append(row)
    return views
