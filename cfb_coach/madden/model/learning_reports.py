"""Read-only explainable reports for expert and personal learning."""
from __future__ import annotations

from typing import Any, Mapping

from cfb_coach.madden.model.expert_film import expert_film_summary, reviewed_expert_snaps
from cfb_coach.madden.model.expert_policy import (
    load_expert_outcome,
    load_expert_policy,
    predict_expert_preferences,
)
from cfb_coach.madden.model.expert_signal import EXPERT_SIGNAL_CAP, load_expert_signal
from cfb_coach.madden.model.learning_sources import (
    VOD_PRIOR_INTERACTION,
    list_evidence,
    source_summary,
)
from cfb_coach.madden.model.personalization import load_personalization


def expert_learning_summary(
    *,
    learning_store: str,
    film_store: str | None = None,
    db: Any = None,
) -> dict[str, Any]:
    sources = source_summary(learning_store)
    film = expert_film_summary(film_store) if film_store else {}
    policy = load_expert_policy(store=learning_store)
    outcome = load_expert_outcome(store=learning_store)
    signal = load_expert_signal(db)
    reviewed = reviewed_expert_snaps(film_store) if film_store else []
    concept_prefs = {}
    if policy:
        # Sample a few common situations for the report.
        for down, distance in ((1, 10), (3, 7), (2, 5)):
            key = f"{down}&{distance}"
            concept_prefs[key] = predict_expert_preferences(
                policy, down=down, distance=distance, level="concept_family",
            )
    label_coverage = {}
    if policy:
        for level, table in (policy.get("levels") or {}).items():
            label_coverage[level] = {
                "labeled_examples": table.get("labeled_examples"),
                "global_effective_n": (table.get("global") or {}).get("effective_n"),
            }
    return {
        "report": "expert-learning",
        "expert_vods": film.get("matches", 0),
        "reviewed_snaps": film.get("reviewed_snaps", len(reviewed)),
        "experts": film.get("experts") or sorted({
            str((row.get("provenance") or {}).get("expert_id"))
            for row in list_evidence(learning_store, category="expert_evidence")
            if (row.get("provenance") or {}).get("expert_id")
        }),
        "label_confidence_and_coverage": label_coverage,
        "expert_concept_preferences_by_situation": concept_prefs,
        "verified_expert_outcome_evidence": {
            "verified_outcome_examples": None if not outcome else outcome.get("verified_outcome_examples"),
            "published_cells": None if not outcome else outcome.get("published_cells"),
            "note": None if not outcome else outcome.get("note"),
        },
        "policy_mode": None if not policy else policy.get("mode"),
        "signal": {
            "mode": None if not signal else signal.get("mode", "shadow"),
            "live_influence": bool(signal and signal.get("live_influence")),
            "cap": EXPERT_SIGNAL_CAP,
            "changes_decisions": bool(signal and signal.get("mode") == "bounded_active"),
        },
        "sources": sources,
        "vod_prior_interaction": VOD_PRIOR_INTERACTION,
        "raw_video_automatically_understood": False,
        "evidence_gaps": _expert_gaps(sources, policy, outcome, film),
    }


def personal_learning_report(
    *,
    learning_store: str,
    opponent: str = "cpu",
    db: Any = None,
) -> dict[str, Any]:
    personal = list_evidence(learning_store, category="personal_evidence")
    if opponent and opponent != "all":
        filtered = [
            row for row in personal
            if str((row.get("situation") or {}).get("opponent_category") or "") == opponent
            or opponent in str((row.get("situation") or {}).get("opponent_category") or "")
        ]
    else:
        filtered = personal
    artifact = load_personalization(store=learning_store, db=db)
    contexts = []
    for key, cell in ((artifact or {}).get("contexts") or {}).items():
        if opponent != "all" and opponent not in key and "unknown" not in key:
            continue
        contexts.append({"context": key, **cell})
    games = sorted({str(row.get("match_id")) for row in filtered if row.get("match_id")})
    return {
        "report": "personal-learning",
        "opponent": opponent,
        "personal_games": len(games),
        "personal_snaps": len(filtered),
        "personalization_weights": contexts[:40],
        "evolution": list(((artifact or {}).get("evolution") or []))[:40],
        "uncertainty_summary": {
            "mean_uncertainty": _mean([c.get("uncertainty") for c in contexts]),
            "conflict_contexts": sum(1 for c in contexts if c.get("conflict")),
        },
        "mode": None if not artifact else artifact.get("mode", "shadow"),
        "fixed_transition_schedule": False,
        "unexecuted_recommendations_excluded": True,
        "evidence_gaps": _personal_gaps(filtered, artifact),
    }


def learning_compare_report(
    *,
    learning_store: str,
    concept: str,
    db: Any = None,
) -> dict[str, Any]:
    concept = str(concept).lower()
    expert_rows = [
        row for row in list_evidence(learning_store, category="expert_evidence")
        if _row_concept(row) == concept
    ]
    personal_rows = [
        row for row in list_evidence(learning_store, category="personal_evidence")
        if _row_concept(row) == concept
    ]
    personalization = load_personalization(store=learning_store, db=db)
    related = {
        key: cell
        for key, cell in ((personalization or {}).get("contexts") or {}).items()
        if key.endswith(f"|{concept}")
    }
    agree = []
    conflict = []
    for key, cell in related.items():
        entry = {
            "context": key,
            "personal_weight": cell.get("personal_weight"),
            "expert_weight": cell.get("expert_weight"),
            "personal_success_rate": cell.get("personal_success_rate"),
            "uncertainty": cell.get("uncertainty"),
        }
        if cell.get("conflict"):
            conflict.append(entry)
        else:
            agree.append(entry)
    signal = load_expert_signal(db)
    return {
        "report": "learning-compare",
        "concept": concept,
        "expert_snap_evidence": len(expert_rows),
        "personal_snap_evidence": len(personal_rows),
        "agree": agree,
        "conflict": conflict,
        "expert_signal_changes_decisions": bool(
            signal and signal.get("mode") == "bounded_active" and signal.get("live_influence")
        ),
        "examples": _compare_examples(expert_rows, personal_rows),
        "evidence_gaps": _compare_gaps(expert_rows, personal_rows, related),
        "strategy_evolution": list(((personalization or {}).get("evolution") or [])),
    }


def _row_concept(row: Mapping[str, Any]) -> str:
    action = row.get("observed_action") or {}
    concept = action.get("concept_family")
    if concept:
        return str(concept).lower()
    play = action.get("play")
    if not play:
        return ""
    from cfb_coach.madden.model.experimental_model import _play_concept
    from cfb_coach.madden.model.football_knowledge import profile_for_play

    return str(profile_for_play(play).get("concept_id") or _play_concept(play) or "").lower()


def _mean(values: list[Any]) -> float | None:
    nums = []
    for value in values:
        try:
            if value is not None:
                nums.append(float(value))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return round(sum(nums) / len(nums), 5)


def _expert_gaps(sources, policy, outcome, film) -> list[str]:
    gaps = []
    if sources.get("counts", {}).get("expert_evidence", 0) == 0:
        gaps.append("no expert evidence rows retained")
    if not policy:
        gaps.append("expert policy not trained")
    if not outcome or not outcome.get("published_cells"):
        gaps.append("insufficient verified expert outcomes for published cells")
    if film and film.get("duplicate_matches"):
        gaps.append("duplicate expert recordings detected; not counted independently")
    if film is not None and film.get("reviewed_snaps", 0) == 0:
        gaps.append("no human-reviewed expert snaps yet")
    return gaps


def _personal_gaps(rows, artifact) -> list[str]:
    gaps = []
    if not rows:
        gaps.append("no verified personal executions for this opponent filter")
    if not artifact:
        gaps.append("personalization artifact not fitted")
    elif int(artifact.get("personal_rows") or 0) < 8:
        gaps.append("personal sample too small for strong conclusions")
    return gaps


def _compare_gaps(expert_rows, personal_rows, related) -> list[str]:
    gaps = []
    if not expert_rows:
        gaps.append("no expert evidence for this concept")
    if not personal_rows:
        gaps.append("no personal evidence for this concept")
    if expert_rows and personal_rows and not related:
        gaps.append("sources exist but no overlapping situational context cells")
    return gaps


def _compare_examples(expert_rows, personal_rows) -> list[dict[str, Any]]:
    examples = []
    for row in expert_rows[:3]:
        examples.append({
            "source": "expert",
            "match_id": row.get("match_id"),
            "action": row.get("observed_action"),
            "situation": row.get("situation"),
            "why": "demonstrated expert choice; not a claim about unchosen plays",
        })
    for row in personal_rows[:3]:
        examples.append({
            "source": "personal",
            "match_id": row.get("match_id"),
            "action": row.get("observed_action"),
            "outcome": row.get("verified_outcome"),
            "why": "verified user execution and labeled outcome",
        })
    return examples
