"""Contextual expert playcalling policy and verified-outcome models.

An observed expert choice is a demonstrated action, not proof that every
unchosen play would have failed. Expert execution ability is not conflated
with offensive concept quality. Labels are confidence-aware.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.experimental_model import _dd_bucket, _play_concept, _play_family
from cfb_coach.madden.model.football_knowledge import profile_for_play
from cfb_coach.madden.model.learning_sources import (
    ensure_store,
    fingerprint_payload,
    list_evidence,
    utc_now,
)

POLICY_SCHEMA = "madden.expert.policy.v1"
OUTCOME_SCHEMA = "madden.expert.outcome.v1"
POLICY_VERSION = "expert_policy.v1"
OUTCOME_VERSION = "expert_outcome.v1"
PRIOR_STRENGTH = 4.0
MIN_OUTCOME_N = 6


def _situation_key(row: Mapping[str, Any]) -> str:
    sit = row.get("situation") or {}
    return "|".join([
        _dd_bucket(sit.get("down"), sit.get("distance")),
        str(sit.get("opponent_type") or "unknown"),
        str(sit.get("competitive_mode") or (row.get("provenance") or {}).get("competitive_mode") or "unknown"),
        str((row.get("provenance") or {}).get("game_version") or "madden27"),
        str((row.get("provenance") or {}).get("patch") or "unspecified"),
    ])


def _action_labels(row: Mapping[str, Any]) -> dict[str, str | None]:
    action = row.get("observed_action") or {}
    play = action.get("play")
    formation = action.get("formation")
    concept = action.get("concept_family") or (
        profile_for_play(play).get("concept_id") if play else None
    ) or (_play_concept(play) if play else None)
    family = action.get("play_action_family") or (_play_family(play) if play else None)
    formation_family = action.get("formation_family")
    if formation and not formation_family:
        formation_family = str(formation).split()[0].lower() if str(formation).strip() else None
    adjustment = None
    adj = action.get("offensive_adjustments")
    if isinstance(adj, str) and adj not in ("", "unknown"):
        adjustment = adj
    elif isinstance(adj, Mapping) and adj.get("id"):
        adjustment = str(adj.get("id"))
    return {
        "concept_family": None if concept in (None, "", "unknown") else str(concept),
        "formation_family": None if formation_family in (None, "", "unknown") else str(formation_family),
        "play": None if play in (None, "", "unknown") else str(play),
        "run_pass_family": None if family in (None, "", "unknown") else str(family),
        "adjustment": adjustment,
    }


def _confidence(row: Mapping[str, Any]) -> float:
    raw = row.get("confidence")
    try:
        value = 0.5 if raw is None else float(raw)
    except (TypeError, ValueError):
        value = 0.5
    return max(0.05, min(1.0, value))


def prepare_expert_dataset(
    learning_store: str | Path,
    *,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Materialize expert evidence into a training dataset with confidence weights."""
    rows = [
        row for row in list_evidence(learning_store, category="expert_evidence")
        if row.get("observed_action")
    ]
    examples = []
    for row in rows:
        labels = _action_labels(row)
        if not any(labels.values()):
            continue
        examples.append({
            "evidence_id": row.get("evidence_id"),
            "match_id": row.get("match_id"),
            "expert_id": (row.get("provenance") or {}).get("expert_id"),
            "situation_key": _situation_key(row),
            "situation": row.get("situation") or {},
            "labels": labels,
            "weight": _confidence(row),
            "demonstrated_action": True,
            "unchosen_failure_claim": False,
            "outcome": row.get("verified_outcome"),
            "provenance": row.get("provenance") or {},
            "fingerprint": row.get("fingerprint"),
        })
    # Deduplicate overlapping recordings by fingerprint.
    seen: set[str] = set()
    unique = []
    for example in examples:
        digest = str(example.get("fingerprint") or example.get("evidence_id"))
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(example)
    dataset = {
        "schema": "madden.expert.dataset.v1",
        "prepared_at": utc_now(),
        "n_examples": len(unique),
        "n_matches": len({ex.get("match_id") for ex in unique if ex.get("match_id")}),
        "n_experts": len({ex.get("expert_id") for ex in unique if ex.get("expert_id")}),
        "examples": unique,
        "split_note": (
            "Evaluation must hold out entire matches and expert players, "
            "not random adjacent frames."
        ),
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataset, indent=2, sort_keys=True), encoding="utf-8")
        dataset["path"] = str(path)
    return dataset


def _dirichlet_probs(
    counts: Mapping[str, float],
    *,
    prior: float = PRIOR_STRENGTH,
) -> dict[str, float]:
    labels = sorted(counts)
    if not labels:
        return {}
    total = sum(float(counts[label]) for label in labels) + prior * len(labels)
    return {
        label: round((float(counts[label]) + prior) / total, 6)
        for label in labels
    }


def train_expert_policy(
    learning_store: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    dataset: Mapping[str, Any] | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    """Train a confidence-aware contextual expert policy at several label levels."""
    prepared = dataset or prepare_expert_dataset(learning_store)
    examples = list(prepared.get("examples") or [])
    levels = (
        "concept_family",
        "formation_family",
        "play",
        "run_pass_family",
        "adjustment",
    )
    # situation -> level -> label -> weight
    tables: dict[str, dict[str, dict[str, float]]] = {
        level: defaultdict(lambda: defaultdict(float)) for level in levels
    }
    global_tables: dict[str, dict[str, float]] = {
        level: defaultdict(float) for level in levels
    }
    drive_transitions: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    previous_by_match: dict[str, str | None] = {}
    coverage: dict[str, int] = {level: 0 for level in levels}

    for example in examples:
        sit = str(example.get("situation_key") or "unknown")
        weight = float(example.get("weight") or 0.5)
        labels = example.get("labels") or {}
        match_id = str(example.get("match_id") or "")
        concept = labels.get("concept_family")
        prev = previous_by_match.get(match_id)
        if prev and concept:
            drive_transitions[prev][concept] += weight
        if concept:
            previous_by_match[match_id] = concept
        for level in levels:
            label = labels.get(level)
            if not label:
                continue
            tables[level][sit][label] += weight
            global_tables[level][label] += weight
            coverage[level] += 1

    policy_tables = {}
    for level in levels:
        by_sit = {}
        for sit, counts in tables[level].items():
            by_sit[sit] = {
                "probs": _dirichlet_probs(counts),
                "effective_n": round(sum(counts.values()), 3),
                "labels": sorted(counts),
            }
        policy_tables[level] = {
            "by_situation": by_sit,
            "global": {
                "probs": _dirichlet_probs(global_tables[level]),
                "effective_n": round(sum(global_tables[level].values()), 3),
            },
            "labeled_examples": coverage[level],
        }

    transitions = {
        source: _dirichlet_probs(targets)
        for source, targets in drive_transitions.items()
    }
    artifact = {
        "schema": POLICY_SCHEMA,
        "version": POLICY_VERSION,
        "trained_at": utc_now(),
        "n_examples": len(examples),
        "n_matches": prepared.get("n_matches"),
        "n_experts": prepared.get("n_experts"),
        "levels": policy_tables,
        "drive_concept_transitions": transitions,
        "demonstrated_action_only": True,
        "unchosen_failure_claim": False,
        "expert_skill_not_concept_quality": True,
        "accounts_for": [
            "franchise_vs_ultimate_team_via_competitive_mode",
            "cpu_vs_human_via_opponent_type",
            "game_version_and_patch",
            "situation_context",
        ],
        "mode": "shadow",
        "fingerprint": None,
    }
    artifact["fingerprint"] = fingerprint_payload({
        k: artifact[k] for k in (
            "schema", "version", "n_examples", "levels", "drive_concept_transitions",
        )
    })
    if persist:
        paths = ensure_store(learning_store)
        out_dir = Path(artifact_dir) if artifact_dir else paths["artifacts"] / "expert_policy"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "policy.json"
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        artifact["path"] = str(path)
    return artifact


def train_expert_outcome_model(
    learning_store: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train an outcome model only on sufficiently verified visible results."""
    prepared = dataset or prepare_expert_dataset(learning_store)
    successes: dict[str, float] = defaultdict(float)
    trials: dict[str, float] = defaultdict(float)
    used = 0
    for example in prepared.get("examples") or []:
        outcome = example.get("outcome") or {}
        if not outcome or not outcome.get("visible") or not outcome.get("verified"):
            continue
        labels = example.get("labels") or {}
        concept = labels.get("concept_family")
        if not concept:
            continue
        sit = str(example.get("situation_key") or "unknown")
        key = f"{sit}|{concept}"
        weight = float(example.get("weight") or 0.5)
        result = outcome.get("result")
        success = None
        if isinstance(result, bool):
            success = result
        elif isinstance(result, Mapping):
            success = result.get("success")
        elif str(result).lower() in ("true", "success", "1", "win"):
            success = True
        elif str(result).lower() in ("false", "failure", "0", "loss"):
            success = False
        if success is None:
            continue
        trials[key] += weight
        if success:
            successes[key] += weight
        used += 1

    cells = {}
    for key, n in trials.items():
        if n < MIN_OUTCOME_N:
            continue
        rate = (successes[key] + PRIOR_STRENGTH * 0.5) / (n + PRIOR_STRENGTH)
        cells[key] = {
            "success_rate": round(rate, 5),
            "effective_n": round(n, 3),
            "insufficient": False,
        }
    artifact = {
        "schema": OUTCOME_SCHEMA,
        "version": OUTCOME_VERSION,
        "trained_at": utc_now(),
        "verified_outcome_examples": used,
        "published_cells": len(cells),
        "min_outcome_n": MIN_OUTCOME_N,
        "cells": cells,
        "expert_skill_not_concept_quality": True,
        "mode": "shadow",
        "note": (
            "Outcome rates reflect verified visible results of expert-executed "
            "plays. They do not score unchosen alternatives."
        ),
    }
    artifact["fingerprint"] = fingerprint_payload({
        k: artifact[k] for k in ("schema", "version", "cells", "verified_outcome_examples")
    })
    paths = ensure_store(learning_store)
    out_dir = Path(artifact_dir) if artifact_dir else paths["artifacts"] / "expert_outcome"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "outcome.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    artifact["path"] = str(path)
    return artifact


def load_expert_policy(path: str | Path | None = None, store: str | Path | None = None) -> dict[str, Any] | None:
    if path is not None:
        candidate = Path(path)
    elif store is not None:
        candidate = ensure_store(store)["artifacts"] / "expert_policy" / "policy.json"
    else:
        return None
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_expert_outcome(path: str | Path | None = None, store: str | Path | None = None) -> dict[str, Any] | None:
    if path is not None:
        candidate = Path(path)
    elif store is not None:
        candidate = ensure_store(store)["artifacts"] / "expert_outcome" / "outcome.json"
    else:
        return None
    if not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def predict_expert_preferences(
    policy: Mapping[str, Any],
    *,
    down: Any,
    distance: Any,
    opponent_type: str = "unknown",
    competitive_mode: str = "unknown",
    game_version: str = "madden27",
    patch: str | None = None,
    level: str = "concept_family",
) -> dict[str, Any]:
    sit = "|".join([
        _dd_bucket(down, distance),
        opponent_type or "unknown",
        competitive_mode or "unknown",
        game_version or "madden27",
        patch or "unspecified",
    ])
    level_table = (policy.get("levels") or {}).get(level) or {}
    cell = (level_table.get("by_situation") or {}).get(sit)
    source = "situation"
    if cell is None:
        cell = level_table.get("global") or {"probs": {}, "effective_n": 0}
        source = "global_fallback"
    return {
        "level": level,
        "situation_key": sit,
        "source": source,
        "probs": dict(cell.get("probs") or {}),
        "effective_n": cell.get("effective_n") or 0,
        "demonstrated_action_only": True,
    }


def score_play_under_expert_policy(
    policy: Mapping[str, Any] | None,
    play: str,
    *,
    down: Any,
    distance: Any,
    opponent_type: str = "cpu",
    competitive_mode: str = "unknown",
    game_version: str = "madden27",
    patch: str | None = None,
) -> dict[str, Any]:
    """Return a bounded preference score for an in-book play under the expert policy."""
    if not policy:
        return {
            "delta": 0.0,
            "reasons": ["no expert policy artifact"],
            "effective_n": 0,
            "concept_id": None,
        }
    profile = profile_for_play(play)
    concept = profile.get("concept_id") or _play_concept(play)
    family = _play_family(play)
    concept_pref = predict_expert_preferences(
        policy, down=down, distance=distance, opponent_type=opponent_type,
        competitive_mode=competitive_mode, game_version=game_version, patch=patch,
        level="concept_family",
    )
    family_pref = predict_expert_preferences(
        policy, down=down, distance=distance, opponent_type=opponent_type,
        competitive_mode=competitive_mode, game_version=game_version, patch=patch,
        level="run_pass_family",
    )
    concept_p = float((concept_pref.get("probs") or {}).get(str(concept), 0.0))
    family_p = float((family_pref.get("probs") or {}).get(str(family), 0.0))
    # Convert preference mass into a small centered score in [-1, 1].
    concept_score = 2.0 * concept_p - (2.0 / max(1, len(concept_pref.get("probs") or {concept: 1})))
    family_score = 2.0 * family_p - (2.0 / max(1, len(family_pref.get("probs") or {family: 1})))
    raw = 0.7 * concept_score + 0.3 * family_score
    effective_n = float(concept_pref.get("effective_n") or 0.0)
    # Shrink toward zero when evidence is thin.
    shrink = effective_n / (effective_n + PRIOR_STRENGTH)
    score = raw * shrink
    reasons = []
    if concept_p > 0:
        reasons.append(
            f"experts demonstrated {concept} in this situation "
            f"(p={concept_p:.3f}, n_eff={effective_n:.1f})"
        )
    else:
        reasons.append("no expert demonstration for this concept in-context; global shrink applied")
    return {
        "delta": round(max(-1.0, min(1.0, score)), 5),
        "reasons": reasons,
        "effective_n": effective_n,
        "concept_id": concept,
        "family": family,
        "concept_p": concept_p,
        "family_p": family_p,
        "source": concept_pref.get("source"),
        "demonstrated_action_only": True,
        "unchosen_failure_claim": False,
    }
