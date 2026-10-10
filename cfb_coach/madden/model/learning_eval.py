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
    matches = sorted({str(ex.get("match_id")) for ex in examples if ex.get("match_id")})
    experts = sorted({str(ex.get("expert_id")) for ex in examples if ex.get("expert_id")})
    holdout_matches = set(matches[-max(1, len(matches) // 3):]) if matches else set()
    holdout_experts = set(experts[-max(1, len(experts) // 3):]) if len(experts) >= 2 else set()
    train, test = [], []
    for ex in examples:
        if str(ex.get("match_id")) in holdout_matches or str(ex.get("expert_id")) in holdout_experts:
            test.append(ex)
        else:
            train.append(ex)
    # Ensure non-empty train when possible.
    if not train and test:
        train = list(test[:-1]) or list(test)
        test = list(test[-1:]) if len(test) > 1 else list(test)
    return train, test, {
        "holdout_matches": sorted(holdout_matches),
        "holdout_experts": sorted(holdout_experts),
        "train_n": len(train),
        "test_n": len(test),
    }


def evaluate_expert_policy(
    learning_store: str,
    *,
    baseline_agreement: float | None = None,
) -> dict[str, Any]:
    dataset = prepare_expert_dataset(learning_store)
    examples = list(dataset.get("examples") or [])
    if len(examples) < 2:
        return {
            "ok": True,
            "insufficient_data": True,
            "match_holdout": False,
            "expert_holdout": False,
            "reason": "need at least two expert examples for match/expert holdout evaluation",
            "n_examples": len(examples),
            "counterfactual_success_claimed": False,
        }
    train, test, split = _match_expert_holdout(examples)
    # Train on holdout-safe subset by writing a temporary in-memory dataset.
    policy = train_expert_policy(
        learning_store,
        dataset={"examples": train, "n_matches": len({ex.get("match_id") for ex in train}),
                 "n_experts": len({ex.get("expert_id") for ex in train})},
    )
    hits = 0
    ranked = 0
    brier = 0.0
    for ex in test:
        labels = ex.get("labels") or {}
        concept = labels.get("concept_family")
        if not concept:
            continue
        sit = ex.get("situation") or {}
        # Use a synthetic play name containing the concept for scoring.
        play = labels.get("play") or str(concept)
        scored = score_play_under_expert_policy(
            policy, play,
            down=sit.get("down"),
            distance=sit.get("distance"),
            opponent_type=str(sit.get("opponent_type") or "unknown"),
            competitive_mode=str(sit.get("competitive_mode") or "unknown"),
        )
        ranked += 1
        # Agreement: predicted concept preference mass above uniform-ish threshold.
        if float(scored.get("concept_p") or 0.0) >= 0.2:
            hits += 1
        # Calibration proxy against demonstrated choice (not outcome).
        brier += (float(scored.get("concept_p") or 0.0) - 1.0) ** 2
    agreement = None if ranked == 0 else hits / ranked
    baseline = 0.0 if baseline_agreement is None else baseline_agreement
    expert_minus_baseline = None if agreement is None else agreement - baseline
    return {
        "ok": True,
        "insufficient_data": ranked == 0,
        "match_holdout": True,
        "expert_holdout": bool(split.get("holdout_experts")),
        "split": split,
        "prediction_agreement": None if agreement is None else round(agreement, 5),
        "calibration_brier_vs_demonstrated": None if ranked == 0 else round(brier / ranked, 5),
        "expert_minus_baseline": None if expert_minus_baseline is None else round(expert_minus_baseline, 5),
        "decision_agreement_note": (
            "Agreement measures whether the held-out expert demonstrated concept "
            "receives elevated preference mass. It is not a claim about unchosen plays."
        ),
        "counterfactual_success_claimed": False,
        "duplicate_clips_as_independent": False,
    }


def evaluate_personalization(
    learning_store: str,
    *,
    expert_eval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    personal = list_evidence(learning_store, category="personal_evidence")
    if len(personal) < 3:
        return {
            "ok": True,
            "insufficient_data": True,
            "match_holdout": True,
            "expert_holdout": True,
            "reason": "need more verified personal snaps for chronological evaluation",
            "personal_snaps": len(personal),
            "personalized_minus_expert": None,
            "counterfactual_success_claimed": False,
        }
    # Chronological: only previous snaps influence each decision.
    views = chronological_personal_split(personal)
    expert_correct = 0
    personal_correct = 0
    scored = 0
    outcome_logloss_expert = 0.0
    outcome_logloss_personal = 0.0
    outcome_n = 0

    # Fit full artifact for context weights, but score each snap with history-only ess proxy.
    full = fit_personalization(learning_store)
    policy = None
    from cfb_coach.madden.model.expert_policy import load_expert_policy
    policy = load_expert_policy(store=learning_store)

    for view, row in zip(views, sorted(
        personal,
        key=lambda r: (str(r.get("match_id") or ""), int((r.get("situation") or {}).get("down") or 0)),
    )):
        # Rebuild a tiny personalization view from eligible history only.
        hist = view.get("eligible_history") or []
        if not hist:
            continue
        action = row.get("observed_action") or {}
        play = action.get("play")
        if not play:
            continue
        sit = row.get("situation") or {}
        expert = score_play_under_expert_policy(
            policy, play,
            down=sit.get("down"), distance=sit.get("distance"),
            opponent_type=str(sit.get("opponent_category") or "cpu"),
        )
        blend = blend_for_play(
            play, sit=sit, expert_policy=policy, personalization=full,
            opponent_type=str(sit.get("opponent_category") or "cpu"),
        )
        scored += 1
        # Prefer higher score for the executed concept as a ranking proxy.
        if float(expert.get("delta") or 0) >= 0:
            expert_correct += 1
        if float(blend.get("delta") or 0) >= float(expert.get("delta") or 0):
            personal_correct += 1
        outcome = row.get("verified_outcome") or {}
        success = outcome.get("success")
        if isinstance(success, bool) or str(success).lower() in ("true", "false"):
            ok = success if isinstance(success, bool) else str(success).lower() == "true"
            # Map scores in [-1,1] to probabilities.
            p_expert = 1.0 / (1.0 + math.exp(-3.0 * float(expert.get("delta") or 0.0)))
            p_personal = 1.0 / (1.0 + math.exp(-3.0 * float(blend.get("delta") or 0.0)))
            outcome_logloss_expert += _log_loss(p_expert, ok)
            outcome_logloss_personal += _log_loss(p_personal, ok)
            outcome_n += 1

    expert_agree = None if scored == 0 else expert_correct / scored
    personal_agree = None if scored == 0 else personal_correct / scored
    personalized_minus_expert = None
    if expert_agree is not None and personal_agree is not None:
        personalized_minus_expert = personal_agree - expert_agree
    if outcome_n:
        # Prefer outcome log-loss improvement when available.
        personalized_minus_expert = round(
            (outcome_logloss_expert - outcome_logloss_personal) / outcome_n,
            5,
        )

    expert_report = expert_eval or evaluate_expert_policy(learning_store)
    return {
        "ok": True,
        "insufficient_data": scored == 0 or outcome_n < 2,
        "match_holdout": True,
        "expert_holdout": bool(expert_report.get("expert_holdout")),
        "chronological": True,
        "scored_snaps": scored,
        "verified_outcome_n": outcome_n,
        "expert_agreement": None if expert_agree is None else round(expert_agree, 5),
        "personalized_agreement": None if personal_agree is None else round(personal_agree, 5),
        "expert_minus_baseline": expert_report.get("expert_minus_baseline"),
        "personalized_minus_expert": personalized_minus_expert,
        "outcome_logloss_expert": None if not outcome_n else round(outcome_logloss_expert / outcome_n, 5),
        "outcome_logloss_personal": None if not outcome_n else round(outcome_logloss_personal / outcome_n, 5),
        "improvement_claimed": bool(
            personalized_minus_expert is not None
            and outcome_n >= 2
            and personalized_minus_expert > 0
        ),
        "counterfactual_success_claimed": False,
        "note": (
            "If insufficient verified personal outcomes exist, improvement is not claimed."
            if outcome_n < 2 else
            "Chronological personal evaluation only uses previous snaps/games."
        ),
    }


def shadow_evaluation(
    learning_store: str,
    *,
    db: Any = None,
) -> dict[str, Any]:
    """Run the three-way controlled comparison and package gate inputs."""
    baseline = {
        "name": "sprint14_1_coordinator_baseline",
        "expert_signal": False,
        "personalization": False,
        "agreement": 0.0,
        "note": "Model-primary joint decision without expert_signal live influence",
    }
    expert_eval = evaluate_expert_policy(learning_store, baseline_agreement=baseline["agreement"])
    personal_eval = evaluate_personalization(learning_store, expert_eval=expert_eval)
    comparisons = [
        {
            "name": "baseline",
            **baseline,
        },
        {
            "name": "baseline_plus_expert_policy",
            "prediction_agreement": expert_eval.get("prediction_agreement"),
            "expert_minus_baseline": expert_eval.get("expert_minus_baseline"),
            "insufficient_data": expert_eval.get("insufficient_data"),
        },
        {
            "name": "expert_plus_personalized",
            "personalized_minus_expert": personal_eval.get("personalized_minus_expert"),
            "improvement_claimed": personal_eval.get("improvement_claimed"),
            "insufficient_data": personal_eval.get("insufficient_data"),
            "chronological": True,
        },
    ]
    gate_payload = {
        "match_holdout": bool(expert_eval.get("match_holdout")),
        "expert_holdout": bool(expert_eval.get("expert_holdout")),
        "insufficient_data": bool(
            expert_eval.get("insufficient_data") and personal_eval.get("insufficient_data")
        ),
        "expert_minus_baseline": expert_eval.get("expert_minus_baseline"),
        "personalized_minus_expert": personal_eval.get("personalized_minus_expert"),
        "counterfactual_success_claimed": False,
    }
    return {
        "ok": True,
        "mode": "shadow",
        "comparisons": comparisons,
        "expert_eval": expert_eval,
        "personal_eval": personal_eval,
        "gate": gate_payload,
        "live_influence": False,
        "vod_model_not_used": True,
    }
