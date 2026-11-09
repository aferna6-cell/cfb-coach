"""Bounded model-scored individual hot-route recommendation on a chosen stock play.

This is NOT a saved macro. It never invents an Xbox receiver icon or modifies
game controls. Existing route-aware schema, first-party player identity, and
owner-confirmed Madden action readiness are strict gates. No learned causal
route payoff is claimed; transformed plays are scored by structural
extrapolation and football matchup priors only when enabled.
"""
from __future__ import annotations

from typing import Any, Mapping


def maybe_add_single_route(
    joint: Mapping[str, Any], *,
    route_ctx: Any,
    sit: Any,
    opponent_id: str,
    remaining_budget_ms: float,
    confirmations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(joint)
    # The main model's complete play choice remains authoritative. Only
    # NO_ADJUSTMENT is considered for supplemental model route search.
    if result.get("kind") != "none" or remaining_budget_ms < 10.0:
        return result
    art = getattr(route_ctx, "art", None)
    if not getattr(route_ctx, "applied", False) or art is None:
        return result
    if not getattr(art, "residual_active", False) and not getattr(route_ctx, "priors", False):
        return result
    identity = result.get("joint") or {}
    form, play = identity.get("formation"), identity.get("play")
    if not isinstance(form, str) or not isinstance(play, str) or not form or not play:
        return result

    from cfb_coach.madden.adjustments import executable as ex
    from cfb_coach.madden.model import offense_decision as od
    from cfb_coach.madden.model import offense_play_features as pf
    from cfb_coach.madden.model import offense_structural_model as sm

    structure = sm.structure_any(form, play)
    eligible = set(pf.eligible_hot_route_targets(structure))
    known = {
        slot for slot, _, tier in structure.routes
        if tier in ("diagram_verified", "in_game_verified")
    }
    if len(eligible & known) == 0 or not eligible.issubset(known):
        # Source-named or anonymous geometry is useful for QB reads, NOT
        # verified Xbox receiver targeting or complete route transformations.
        return result

    extras = getattr(sit, "extras", None) or {}
    remaining_play_clock = extras.get("play_clock_seconds")
    decision_sit = od.Situation(
        down=getattr(sit, "down", None),
        distance=getattr(sit, "distance", None),
        yardline=getattr(sit, "yardline", None),
        opponent_id=opponent_id,
        opponent_type="cpu" if str(opponent_id).lower().startswith("cpu") else "human",
        mix=dict(getattr(route_ctx, "mix", {}) or {}),
        play_clock_remaining=remaining_play_clock,
    )
    # Bounded adjunct to the ALREADY selected play: no second 306-play rank,
    # no macros and no automatic action. Owner-confirmed action options only.
    decision = od.decide(
        art, decision_sit, [(form, play)],
        confirmations=confirmations,
        mode="live", allow_adjustments=True,
        priors=bool(getattr(route_ctx, "priors", False)),
        learned_cells=getattr(route_ctx, "learned", None),
        adjust_top=1, beam_width=1, pool_size=4,
        budget_ms=min(12.0, max(2.0, remaining_budget_ms - 4.0)),
    )
    if (not decision.executable_now or
            len(decision.actions) != 1 or
            decision.actions[0].action_id != "off.hot_route"):
        return result
    action = decision.actions[0]
    slot = action.params.get("target")
    route = action.params.get("route")
    if slot not in eligible or slot not in known or not isinstance(route, str):
        return result
    specification = ex.registry()["off.hot_route"]
    validation = ex.validate_plan(
        (action,), ex.PlayContext(
            side="offense", formation=form, play=play, play_type="pass",
            eligible_receivers=structure.skill_slots,
            play_clock_remaining=remaining_play_clock,
        ), confirmations=confirmations, require_live=True,
    )
    if validation:
        return result
    buttons = ex.render_inputs(specification, action.params)
    ident = f"model_hot_route:{slot}:{route}"
    payload = {
        "id": ident, "kind": "hot_route",
        "target": slot, "route": route,
        "label": f"Hot route {slot} → {route}",
        "buttons": buttons,
        "why": "model-scored situational route change; owner manually executes",
        "sources": list(specification.sources),
        "evidence": "structural_extrapolation_not_verified_open_rate",
        "not_a_macro": True,
        "verified_action_options": True,
    }
    result.update(
        kind="adjustment", id=ident, macro=None, adjustment=payload,
        reason=payload["why"], no_action_compared=True,
        scores_are="model_structural_extrapolation_not_learned_action_effect",
    )
    result["joint"] = dict(identity, plan_id=ident, displaced_baseline=True)
    result["individual_route_audit"] = {
        "stock_probability": decision.stock_probability,
        "modified_probability": decision.probability,
        "route_search_ms": decision.latency_ms,
        "selected": ident,
        "receiver_identity_tier": "first_party_verified",
    }
    return result
