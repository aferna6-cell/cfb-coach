"""Auditable guard for no-adjustment play overrides.

The joint optimizer is free to pick a different play, but a tiny bonus from
strategy/knowledge should not silently displace an independently sampled model
anchor when neither play is adjusted. The margin is a policy proxy, NOT an
estimated causal advantage or true conversion probability.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

NO_ACTION_OVERRIDE_MARGIN = 0.025


def review_override(
    *,
    chosen: Mapping[str, Any],
    anchor: Mapping[str, Any],
    representatives: Sequence[Mapping[str, Any]],
    margin: float = NO_ACTION_OVERRIDE_MARGIN,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    anchor_key = (str(anchor.get("formation")), str(anchor.get("play")))
    selected_key = (str(chosen.get("formation")), str(chosen.get("play")))
    anchor_rep = next(
        (r for r in representatives
         if (str(r.get("formation")), str(r.get("play"))) == anchor_key),
        None,
    )
    selected_score = float(chosen.get("joint_score") or 0.0)
    anchor_score = float(anchor_rep.get("joint_score") or 0.0) if anchor_rep else None
    gain = round(selected_score - anchor_score, 6) if anchor_score is not None else None
    plan = chosen.get("plan") or {}
    no_adjustment = plan.get("plan_id") == "NO_ADJUSTMENT"
    changed = selected_key != anchor_key
    # Apply only to genuine play-only changes. Preserve meaningful joint
    # advantages and legitimate action plans. Nothing changes if anchor
    # is not legal or not represented.
    restrained = bool(
        changed and no_adjustment and anchor_rep is not None
        and gain is not None and gain < margin
    )
    output = anchor_rep if restrained else chosen
    audit = {
        "anchor": list(anchor_key),
        "unrestricted": list(selected_key),
        "selected": [str(output.get("formation")), str(output.get("play"))],
        "unrestricted_score": selected_score,
        "anchor_joint_score": anchor_score,
        "anchor_components": dict(anchor_rep.get("score_components") or {}) if anchor_rep else None,
        "unrestricted_components": dict(chosen.get("score_components") or {}),
        "selected_components": dict(output.get("score_components") or {}),
        "joint_gain": gain,
        "margin": margin,
        "no_adjustment_override": changed and no_adjustment,
        "guard_kept_anchor": restrained,
        "reason": (
            "no-adjustment override under conservative evidence margin"
            if restrained else (
                "joint advantage sufficient or sourced action"
                if changed else "same play as first-stage anchor"
            )
        ),
        "score_claim": "model_proxy_not_counterfactual_outcome",
    }
    return output, audit
