"""Controlled evaluation of expert policy and personalization.

Comparisons:
1. Sprint 14.1 coordinator baseline
2. Baseline plus expert policy
3. Expert policy plus personalized adaptation

Splits are by entire matches and expert players. Personal gameplay uses
chronological evaluation only. Counterfactual success for unexecuted plays
is never claimed.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.expert_policy import (
    prepare_expert_dataset,
    score_play_under_expert_policy,
    train_expert_policy,
)
from cfb_coach.madden.model.learning_sources import list_evidence
from cfb_coach.madden.model.personalization import (
    chronological_personal_split,
    fit_personalization,
    blend_for_play,
)


def _log_loss(prob: float, success: bool) -> float:
    p = min(1.0 - 1e-6, max(1e-6, prob))
    return -math.log(p if success else (1.0 - p))


def _match_expert_holdout(
    examples: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    """Strict expert AND match holdout. Never train on a test example."""
    matches = sorted({str(x.get("match_id")) for x in examples if x.get("match_id")})
    experts = sorted({str(x.get("expert_id")) for x in examples if x.get("expert_id")})
    meta = {
        "holdout_matches": [], "holdout_experts": [],
        "train_n": 0, "test_n": 0, "disjoint": False,
        "reason": "insufficient independent experts and matches",
    }
    if len(experts) < 2 or len(matches) < 2:
        return [], [], meta
    for expert_id in reversed(experts):
        held_matches = {
            str(x.get("match_id")) for x in examples
            if x.get("expert_id") == expert_id and x.get("match_id")
        }
        train = [
            x for x in examples
            if x.get("expert_id") != expert_id
            and str(x.get("match_id")) not in held_matches
            and x.get("expert_id") and x.get("match_id")
        ]
        test = [
            x for x in examples
            if x.get("expert_id") == expert_id and str(x.get("match_id")) in held_matches
        ]
        if train and test:
            train_ids = {str(x.get("evidence_id")) for x in train}
            test_ids = {str(x.get("evidence_id")) for x in test}
            train_matches = {str(x.get("match_id")) for x in train}
            test_matches = {str(x.get("match_id")) for x in test}
            train_experts = {str(x.get("expert_id")) for x in train}
            test_experts = {str(x.get("expert_id")) for x in test}
            disjoint = not (
                train_ids & test_ids or train_matches & test_matches
                or train_experts & test_experts
            )
            if disjoint:
                return train, test, {
                    "holdout_matches": sorted(test_matches),
                    "holdout_experts": sorted(test_experts),
                    "train_n": len(train), "test_n": len(test),
                    "disjoint": True, "reason": None,
                }
    return [], [], meta


def evaluate_expert_policy(
    learning_store: str,
    *,
    baseline_agreement: float | None = None,
) -> dict[str, Any]:
    """Hold out entire matches and players; use a pure, in-memory evaluation fit."""
    from cfb_coach.madden.model.expert_policy import predict_expert_preferences

    examples = list(prepare_expert_dataset(learning_store).get("examples") or [])
    train, test, split = _match_expert_holdout(examples)
    valid_split = bool(split["disjoint"] and train and test)
    insufficient = not valid_split or len(train) < 20 or len(test) < 10
    if not valid_split:
        return {
            "ok": True, "insufficient_data": True,
            "match_holdout": False, "expert_holdout": False,
            "split": split, "n_examples": len(examples),
            "reason": "no disjoint expert-and-match holdout split",
            "coordinator_baseline_measured": False,
            "expert_minus_baseline": None,
            "counterfactual_success_claimed": False,
        }
    # persist=False is critical: evaluation MUST NOT replace policy.json.
    policy = train_expert_policy(
        learning_store, persist=False,
        dataset={
            "examples": train,
            "n_matches": len({x.get("match_id") for x in train}),
            "n_experts": len({x.get("expert_id") for x in train}),
        },
    )
    concept_labels = set((policy.get("levels") or {}).get("concept_family", {}).get("global", {}).get("probs", {}))
    n_labels = max(1, len(concept_labels))
    n = 0
    correct = 0
    model_loss = 0.0
    uniform_loss = 0.0
    out_of_vocab = 0
    for ex in test:
        concept = (ex.get("labels") or {}).get("concept_family")
        if not concept:
            continue
        sit = ex.get("situation") or {}
        provenance = ex.get("provenance") or {}
        pred = predict_expert_preferences(
            policy, down=sit.get("down"), distance=sit.get("distance"),
            opponent_type=str(sit.get("opponent_type") or "unknown"),
            competitive_mode=str(sit.get("competitive_mode") or "unknown"),
            game_version=str(provenance.get("game_version") or "madden27"),
            patch=provenance.get("patch"),
            level="concept_family",
        )
        probs = pred.get("probs") or {}
        choice = float(probs.get(str(concept), 0.0))
        if str(concept) not in concept_labels:
            out_of_vocab += 1
        if probs and max(probs, key=probs.get) == str(concept):
            correct += 1
        model_loss += -math.log(max(1e-6, choice))
        uniform_loss += math.log(n_labels)
        n += 1
    real_test = sum(
        1 for x in test
        if (x.get("provenance") or {}).get("permission_status") == "user_authorized_local"
    )
    return {
        "ok": True,
        "insufficient_data": insufficient or n < 10 or real_test != len(test),
        "match_holdout": valid_split,
        "expert_holdout": valid_split,
        "split": split,
        "n_examples": len(examples),
        "scored_test_examples": n,
        "held_out_real_authorized_examples": real_test,
        "held_out_fixture_examples": len(test) - real_test,
        "top1_expert_choice_accuracy": None if n == 0 else round(correct / n, 5),
        "expert_choice_logloss": None if n == 0 else round(model_loss / n, 5),
        "uniform_choice_logloss": None if n == 0 else round(uniform_loss / n, 5),
        "unseen_concepts": out_of_vocab,
        "prediction_agreement": None if n == 0 else round(correct / n, 5),
        "coordinator_baseline_measured": False,
        "expert_minus_baseline": None,
        "uniform_comparison_only": True,
        "counterfactual_success_claimed": False,
        "evaluation_artifact_written": False,
        "note": "A uniform concept baseline is NOT the current coordinator's actual replay baseline.",
    }


def evaluate_personalization(
    learning_store: str,
    *,
    expert_eval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Use only earlier verified personal evidence for each shadow comparison."""
    personal = list_evidence(learning_store, category="personal_evidence")
    expert_rows = list_evidence(learning_store, category="expert_evidence")
    from cfb_coach.madden.model.expert_policy import load_expert_policy
    policy = load_expert_policy(store=learning_store)
    views = chronological_personal_split(personal)
    valid_views = 0
    deltas = []
    known_outcomes = 0
    for view in views:
        row = view.get("row") or {}
        history = view.get("eligible_history") or []
        action = row.get("observed_action") or {}
        if not action.get("play") or not history:
            continue
        # Each fit uses strictly historical verified executions, never the full file.
        past = fit_personalization(
            learning_store, expert_policy=policy, personal_rows=history,
            expert_rows=expert_rows, persist=False,
        )
        sit = row.get("situation") or {}
        opponent = str(sit.get("opponent_category") or sit.get("opponent_type") or "unknown")
        expert_only = blend_for_play(
            action["play"], sit=sit, expert_policy=policy,
            personalization=None, opponent_type=opponent,
        )
        personalized = blend_for_play(
            action["play"], sit=sit, expert_policy=policy,
            personalization=past, opponent_type=opponent,
        )
        deltas.append(
            float(personalized.get("delta") or 0.0) -
            float(expert_only.get("delta") or 0.0)
        )
        valid_views += 1
        if (row.get("verified_outcome") or {}).get("success") is not None:
            known_outcomes += 1
    expert_report = expert_eval or evaluate_expert_policy(learning_store)
    return {
        "ok": True,
        # Expert-preference scores are not calibrated success probabilities;
        # until a real held-out coordinator replay exists, no efficacy claim.
        "insufficient_data": True,
        "match_holdout": bool(expert_report.get("match_holdout")),
        "expert_holdout": bool(expert_report.get("expert_holdout")),
        "chronological": True,
        "scored_snaps": valid_views,
        "verified_outcome_n": known_outcomes,
        "mean_personal_minus_expert_preference": (
            None if not deltas else round(sum(deltas) / len(deltas), 5)
        ),
        "expert_agreement": None,
        "personalized_agreement": None,
        "personalized_minus_expert": None,
        "outcome_logloss_expert": None,
        "outcome_logloss_personal": None,
        "improvement_claimed": False,
        "future_personal_rows_used": False,
        "evaluation_artifact_written": False,
        "counterfactual_success_claimed": False,
        "note": (
            "Historical-only shadow preference comparison. Not an outcome model "
            "or held-out coordinator evaluation. Live promotion remains blocked."
        ),
    }


def shadow_evaluation(
    learning_store: str,
    *,
    db: Any = None,
) -> dict[str, Any]:
    """Run a shadow-only comparison; never use placeholder baselines as gates."""
    expert_eval = evaluate_expert_policy(learning_store)
    personal_eval = evaluate_personalization(learning_store, expert_eval=expert_eval)
    baseline = {
        "name": "sprint14_1_coordinator_baseline",
        "expert_signal": False, "personalization": False,
        "agreement": None, "measured": False,
        "note": "No real replay against the currently installed coordinator was run.",
    }
    comparisons = [
        {"name": "baseline", **baseline},
        {
            "name": "baseline_plus_expert_policy",
            "prediction_agreement": expert_eval.get("prediction_agreement"),
            "expert_minus_baseline": None,
            "insufficient_data": expert_eval.get("insufficient_data"),
        },
        {
            "name": "expert_plus_personalized",
            "personalized_minus_expert": None,
            "improvement_claimed": False,
            "insufficient_data": True,
            "chronological": True,
        },
    ]
    gate = {
        "match_holdout": bool(expert_eval.get("match_holdout")),
        "expert_holdout": bool(expert_eval.get("expert_holdout")),
        "insufficient_data": True,
        "coordinator_baseline_measured": False,
        "personalized_outcome_eval_measured": False,
        "real_expert_footage_verified": (
            expert_eval.get("held_out_real_authorized_examples", 0) >= 10
            and not expert_eval.get("insufficient_data")
        ),
        "expert_minus_baseline": None,
        "personalized_minus_expert": None,
        "counterfactual_success_claimed": False,
    }
    return {
        "ok": True, "mode": "shadow",
        "comparisons": comparisons,
        "expert_eval": expert_eval,
        "personal_eval": personal_eval,
        "gate": gate,
        "live_influence": False,
        "vod_model_not_used": True,
        "active_model_artifacts_unchanged": True,
    }
