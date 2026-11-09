"""Defensive V2 joint play + individual-adjustment decision search.

Game-existence policy: Madden 27 EA-documented named adjustments may be proposed
without verifying an Xbox button sequence. This does NOT assume that an effect
prior has been demonstrated by outcomes or that an unspecified player target is
known. No macros; NO ADJUSTMENT competes with every candidate play.

Scores are conservative context-conditioned structural hypotheses until enough
verified executed adjustment outcomes are observed. No hidden pre-snap tell.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import defense_v2_knowledge as knowledge

SCHEMA = "madden.defense.v2.adjustment_search.v1"
# For a purely schematic prior, require notable expected reduction in offensive
# success, concentration of opponent belief, and meaningful marginal utility.
MIN_PRIOR_GAIN = 0.018
MIN_COMBO_GAIN = 0.016
PRIOR_SHRINK = 0.33
MAX_ADJUSTMENTS = 2
MAX_TOP_CALLS = 6
# Individual defender assignments with unverified role or unbound player
# identity must not be fabricated. Global Madden-supported actions are eligible.
CATALOG = Path(__file__).resolve().parents[2] / "data" / "madden27" / "executable_actions.json"


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def _action_options(*, model: Mapping[str, Any], sit: Any, play: str, family: str,
                    belief: Mapping[str, float]) -> list[dict[str, Any]]:
    """Only source-backed EXISTING Madden adjustments with known named effect direction."""
    raw = _catalog()
    if raw.get("schema") != "madden27.executable_actions.v1":
        raise ValueError("unrecognized Madden defensive action catalog")
    info = knowledge.concept_knowledge(play, family=family)
    # Uncertain scheme never gets a zone-specific override.
    zone_like = info["scheme"] in ("cover2", "cover3", "cover4", "cover6", "cover9", "tampa2")
    pass_share = sum(float(belief.get(c, 0)) for c in ("vert", "flood", "cross", "stack"))
    opts = []
    for action in raw.get("actions", []):
        if action.get("side") != "defense" or action.get("kind") != "individual_presnap":
            continue
        if not action.get("sources") or action.get("inputs_grade") == "conflict":
            continue
        src = {s for s in action["sources"] if str(s).startswith("ea-")}
        if not src:
            continue
        action_id = str(action.get("id") or "")
        if action_id == "def.coverage_hub":
            groups = action.get("option_groups") or {}
            for option, effect in (action.get("effects_by_param") or {}).items():
                if option not in (action.get("params") or {}).get("option", {}).get("options", []):
                    continue
                if option.startswith("Smart Zone") and not zone_like:
                    continue
                if option.startswith("Pass Commit") and pass_share < 0.72:
                    continue
                # Deep focus needs a receiver target to avoid inventing one.
                if option == "Deep Zone Focus":
                    continue
                opts.append({
                    "id": action_id + "/" + option, "label": option,
                    "kind": "individual", "effect_prior": effect,
                    "group": "hub:" + str(groups.get(option, option)),
                    "game_exists": True, "source_ids": sorted(src),
                    "scheme": info["scheme"],
                    "effect_evidence": "football_prior_not_verified",
                    "target": None,
                })
        elif action_id == "def.double_team":
            extras = getattr(sit, "extras", None) or {}
            receiver = str(extras.get("opponent_receiver_target") or "").strip() if isinstance(extras, Mapping) else ""
            if not receiver or pass_share < 0.75:
                continue
            opts.append({
                "id": action_id + "@" + receiver, "label": "Double Team " + receiver,
                "kind": "individual", "effect_prior": action.get("effects") or {},
                "group": "coverage_target:" + receiver,
                "game_exists": True, "source_ids": sorted(src),
                "effect_evidence": "football_prior_not_verified",
                "target": receiver,
            })
        elif action_id == "def.hot_route":
            # EA confirms per-player hot routes exist, but the specific Madden
            # defender role and allowable new duty must be known before use.
            extras = getattr(sit, "extras", None) or {}
            defender = str(extras.get("defender_target") or "").strip() if isinstance(extras, Mapping) else ""
            receiver = str(extras.get("receiver_target") or "").strip() if isinstance(extras, Mapping) else ""
            if defender and receiver and info["scheme"] != "unknown":
                # The menu's man match-to-receiver capability is documented by EA;
                # no proven effect direction, so leave as an available (not
                # automatically recommended) human-guided option.
                opts.append({
                    "id": action_id + "/match@" + defender + ">" + receiver,
                    "label": defender + " man-match " + receiver,
                    "kind": "individual", "effect_prior": {},
                    "group": "assignment:" + defender,
                    "game_exists": True, "source_ids": sorted(src),
                    "effect_evidence": "effect_unknown", "target": defender,
                })
    return opts


def _prior_gain(option: Mapping[str, Any], belief: Mapping[str, float]) -> float:
    effects = option.get("effect_prior") or {}
    addressed = effects.get("addresses") or {}
    exposed = effects.get("exposes") or {}
    if effects.get("evidence") not in ("prior", "hypothesis"):
        return 0.0
    net = sum(float(belief.get(concept, 0.0)) *
              (float(addressed.get(concept, 0.0)) - float(exposed.get(concept, 0.0)))
              for concept in v2.CONCEPTS)
    return PRIOR_SHRINK * net


def _learned_delta(model: Mapping[str, Any], context: str, family: str,
                   action_id: str) -> tuple[float | None, int]:
    data = (model.get("adjustment_effects") or {}).get(
        json.dumps((context, family, action_id), separators=(",", ":"))
    ) or {}
    if data.get("n", 0) < 12 or data.get("games", 0) < 2:
        return None, int(data.get("n", 0))
    # This is an observational improvement estimate, NOT a causal effect.
    d = float(data.get("offense_risk_delta", 0.0))
    return max(-0.06, min(0.06, -d)), int(data["n"])


def _gain(option: Mapping[str, Any], model: Mapping[str, Any], *,
          context: str, family: str, belief: Mapping[str, float]) -> tuple[float, str, int]:
    prior = _prior_gain(option, belief)
    measured, n = _learned_delta(model, context, family, str(option["id"]))
    if measured is None:
        return prior, "shrunk_football_prior", 0
    # Combine the observed, adjusted-execution comparison and scheme prior.
    # Never claim causality from confounded historical action choices.
    return 0.65 * measured + 0.35 * prior, "observational_and_prior", n


def choose(*, model: Mapping[str, Any], sit: Any,
           ranked: Sequence[Mapping[str, Any]], belief: Mapping[str, float],
           context: str, allow_combinations: bool = True) -> dict[str, Any]:
    """Bounded joint action optimizer; returns exact play and 0-2 individual actions.

    Every ranked call comes from a pre-validated installed-book selector. A
    candidate with insufficient expected improvement remains NO ADJUSTMENT.
    """
    if not ranked:
        raise ValueError("must pass at least one installed candidate")
    # Calibrate against the actual belief. Unknown is not certainty.
    concentration = max((float(x) for x in belief.values()), default=0)
    baseline = min(ranked, key=lambda r: float(r["defensive_utility_risk"]))
    result = {
        "formation": baseline["formation"], "play": baseline["play"],
        "family": baseline["family"],
        "stock_risk": float(baseline["defensive_utility_risk"]),
        "adjusted_risk": float(baseline["defensive_utility_risk"]),
        "actions": [], "action_id": "NO_ADJUSTMENT",
        "adjustment_needed": False, "source": "model_base",
        "evaluated_calls": min(MAX_TOP_CALLS, len(ranked)),
        "reason": "no individual action cleared conservative opponent and risk thresholds",
        "game_exists": True, "compared_no_adjustment": True,
    }
    if concentration < 0.45:
        result["reason"] = "opponent concept belief diffuse: run stock call"
        return result

    for row in ranked[:MAX_TOP_CALLS]:
        form, play, family = str(row["formation"]), str(row["play"]), str(row["family"])
        base = float(row["defensive_utility_risk"])
        available = _action_options(model=model, sit=sit, play=play, family=family, belief=belief)
        evaluated = []
        for option in available:
            benefit, evidence, n = _gain(option, model, context=context, family=family, belief=belief)
            if not math.isfinite(benefit) or benefit < MIN_PRIOR_GAIN:
                continue
            evaluated.append({**option, "expected_risk_reduction": benefit,
                              "estimate_basis": evidence, "verified_executions": n})
        evaluated.sort(key=lambda d: (-d["expected_risk_reduction"], d["id"]))
        for first in evaluated:
            bundles = [[first]]
            if allow_combinations:
                for second in evaluated:
                    if second["id"] == first["id"] or second["group"] == first["group"]:
                        continue
                    # Model diminishing marginal value for simultaneous
                    # changes. Two adjustments need distinct compatible groups.
                    if 0.55 * second["expected_risk_reduction"] < MIN_COMBO_GAIN:
                        continue
                    bundles.append([first, second])
            for bundle in bundles:
                if len(bundle) > MAX_ADJUSTMENTS:
                    continue
                improvement = bundle[0]["expected_risk_reduction"]
                if len(bundle) > 1:
                    improvement += 0.55 * bundle[1]["expected_risk_reduction"]
                score = base - improvement
                # Stronger conservative margin if two independent actions.
                threshold = 0.015 + (0.012 if len(bundle) > 1 else 0.0)
                if (result["adjusted_risk"] - score) <= threshold:
                    continue
                ids = [o["id"] for o in bundle]
                result.update({
                    "formation": form, "play": play, "family": family,
                    "stock_risk": base, "adjusted_risk": score,
                    "actions": bundle,
                    "action_id": ids[0] if len(ids) == 1 else "joint:" + "+".join(ids),
                    "adjustment_needed": True,
                    "source": ("model_plus_observed_adjustment_effect" if
                               any(o["verified_executions"] for o in bundle)
                               else "model_plus_shrunk_football_prior"),
                    "reason": ("Opponent concept distribution favors a source-documented "
                               "defensive individual adjustment; benefit is an estimate, "
                               "not a proven coverage outcome"),
                })
    return result
