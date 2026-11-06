"""Research-only bridge: real Madden adjustment actions + known play assignments.

Calls the existing Madden 27 executable action registry for validation and
selection. Never changes a live recommendation or claims that a play name
implies a receiver route or defensive responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
from typing import Any, Mapping

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model.play_knowledge import (
    AssignmentChange, KnowledgeError, MacroDefinition, PlayKnowledge,
    apply_macro, defensive_user, offense_read, transform,
)


@dataclass(frozen=True)
class AssignmentDecision:
    formation: str
    play: str
    adjustment: str
    coaching_card: str
    read_advice: str
    user_advice: str
    assignment_summary: str
    evidence_status: str
    status: str


def evaluate_assignment_decision(
    ctx: ex.PlayContext,
    knowledge: PlayKnowledge | None,
    opponent_distribution: Mapping[str, float],
    *,
    loadout: ex.Loadout | None = None,
    confirmations: Mapping[str, Any] | None = None,
    verified_macros: Mapping[str, MacroDefinition] | None = None,
    user_defender: str | None = None,
) -> AssignmentDecision:
    """Select a definite *research* adjustment; render only source-backed facts.

    The registry owns legality, clock, installation, confirmation, and scoring.
    Assignment effects are applied only when the transformation is separately
    verified. A valid coverage-hub edit whose detailed effect is unknown is
    represented as unknown, not as a fabricated zone assignment.
    """
    if knowledge is not None and (knowledge.side, knowledge.formation, knowledge.play) != (
        ctx.side, ctx.formation, ctx.play
    ):
        raise KnowledgeError("play knowledge does not match the actual call")

    rec = ex.recommend(ctx, opponent_distribution, loadout=loadout,
                       confirmations=confirmations, mode="live")
    choice = rec.chosen
    label = "NO ADJUSTMENT"
    status = "stock"
    post = knowledge
    assignment_summary = "PLAY ASSIGNMENTS UNKNOWN"
    if knowledge is not None:
        assignment_summary = f"{knowledge.completeness}: {len(knowledge.jobs)} sourced player jobs"

    if choice.spec.kind != "stock":
        # The registry selected a legally executable action. There is a second,
        # independent question: is its exact assignment transformation known?
        label = choice.spec.name
        if choice.params.get("target"):
            label += f": {choice.params['target']}"
        option = next((str(choice.params[k]) for k in choice.spec.params if k in choice.params), None)
        if option:
            label += f" -> {option}"
        status = "action_confirmed; detailed_effect_unknown"
        if post is not None:
            try:
                if choice.spec.id == "off.hot_route":
                    params = choice.params
                    post = transform(post, [AssignmentChange(
                        method="hot_route", target=params["target"], kind="route",
                        task=str(params["route"]).lower(), action_id=choice.spec.id,
                        user_verified=True, source="confirmed direct Xbox hot route",
                    )], eligible_targets=ctx.eligible_receivers)
                    status = "action_confirmed; assignment_change_known"
                elif choice.spec.id == "def.hot_route":
                    # Defensive assignment menu values are not a normalized
                    # taxonomy yet; never assume a submenu option's effect.
                    pass
                elif choice.spec.kind == "saved_custom":
                    macro = (verified_macros or {}).get(choice.spec.name)
                    if macro is not None:
                        post = apply_macro(post, macro,
                                           eligible_targets=ctx.eligible_receivers if ctx.side == "offense" else ctx.defenders)
                        status = "macro_confirmed; assignment_changes_known"
            except (KeyError, KnowledgeError):
                # The action may execute, but we don't know every structural
                # consequence. Do not change our local play representation.
                post = knowledge
                status = "action_confirmed; detailed_effect_unknown"

    read = offense_read(post) if post is not None else "READ UNKNOWN"
    user = defensive_user(post, user_defender) if post is not None and user_defender else "USER RESPONSIBILITY UNKNOWN"
    return AssignmentDecision(
        formation=ctx.formation, play=ctx.play, adjustment=label,
        coaching_card=ex.coaching_card(rec, ctx, loadout=loadout),
        read_advice=read, user_advice=user,
        assignment_summary=assignment_summary,
        evidence_status="verified_play_jobs" if post and post.completeness == "fully_verified" else
                        "partial_play_jobs" if post and post.jobs else "catalog_only",
        status=status,
    )


@dataclass(frozen=True)
class ScoredPlan:
    """Complete play + optional adjustment plan scored by a research model.

    The score is supplied, not invented by this module. Adjustments may be
    individual, saved macros, or legal combinations thereof.
    """
    actions: tuple[ex.PlannedAction, ...]
    model_score: float
    evidence: str


@dataclass(frozen=True)
class PlanSelection:
    actions: tuple[ex.PlannedAction, ...]
    label: str
    model_score: float
    evidence: str


def select_scored_plan(ctx: ex.PlayContext, stock_score: float,
                       proposed: Sequence[ScoredPlan], *,
                       loadout: ex.Loadout | None = None,
                       confirmations: Mapping[str, Any] | None = None,
                       min_gain: float = 0.01) -> PlanSelection:
    """Return a decisive choice from pre-scored plans, safety-gated by Madden.

    This does not estimate that any adjustment is better: it consumes scores
    provided by an outside model/evaluator and refuses invalid combinations.
    Never call from live coaching without an approved, validated model.
    """
    if not math.isfinite(stock_score) or min_gain < 0:
        raise KnowledgeError("invalid stock score or minimum gain")
    valid: list[ScoredPlan] = []
    for candidate in proposed:
        if not candidate.actions or not math.isfinite(candidate.model_score):
            continue
        if not candidate.evidence or candidate.model_score < stock_score + min_gain:
            continue
        if ex.validate_plan(candidate.actions, ctx, loadout=loadout,
                            confirmations=confirmations, require_live=True):
            continue
        valid.append(candidate)
    if not valid:
        return PlanSelection((), "NO ADJUSTMENT", stock_score, "stock_play")
    valid.sort(key=lambda c: (-c.model_score,
                              tuple((p.action_id, tuple(sorted(p.params.items()))) for p in c.actions)))
    winner = valid[0]
    # Display exactly the Madden registry action identifiers and selected
    # parameters. No invented on-screen names or unspecified submenu steps.
    label = " + ".join(
        p.action_id + (" (" + ", ".join(f"{k}={v}" for k, v in sorted(p.params.items())) + ")"
                       if p.params else "") for p in winner.actions
    )
    return PlanSelection(winner.actions, label, winner.model_score, winner.evidence)