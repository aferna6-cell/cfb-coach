"""Defensive V2 offline playbook designer.

Scores stock formations by their best learned coverage options across scenarios;
does not install, confirm availability in Madden's editor, or modify live state.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import defense_v2_pilot as v2

CONTEXT_WEIGHTS = {"early": 0.40, "short": 0.20, "medium": 0.12,
                   "long": 0.20, "unknown": 0.08}
MAX_FORMATIONS = 15


def _risk(model: Mapping[str, Any], context: str, form: str, play: str,
          family: str, belief: Mapping[str, float]) -> float:
    success = sum(
        float(belief[c]) * v2.offense_risk(model, context, form, play, family, c)[0]
        for c in v2.CONCEPTS
    )
    return v2.defensive_utility_risk(
        success, v2.auxiliary_risk(model, context, family)
    )


# Pregame scenario priors are deliberately weak and explicit. The opponent
# distribution still enters each cell, and the utility/outcome model supplies
# the candidate-play predictions. No mandatory formation-type quotas.
SCENARIO_CONCEPTS = {
    "early": {"run": 0.36, "rpo": 0.20, "cross": 0.14, "vert": 0.09,
              "flood": 0.10, "stack": 0.06, "scram": 0.05},
    "short": {"run": 0.57, "rpo": 0.16, "cross": 0.10, "vert": 0.04,
              "flood": 0.06, "stack": 0.04, "scram": 0.03},
    "medium": {"run": 0.20, "rpo": 0.13, "cross": 0.24, "vert": 0.15,
               "flood": 0.15, "stack": 0.08, "scram": 0.05},
    "long": {"run": 0.05, "rpo": 0.04, "cross": 0.26, "vert": 0.27,
             "flood": 0.22, "stack": 0.11, "scram": 0.05},
    "unknown": {c: 1 / len(v2.CONCEPTS) for c in v2.CONCEPTS},
}


def _scenario_probs(ctx: str, belief: Mapping[str, float]) -> dict[str, float]:
    # 60% situation-specific prior, 40% actual opponent belief; both tracked.
    base = SCENARIO_CONCEPTS[ctx]
    total = {c: 0.6 * base[c] + 0.4 * float(belief.get(c, 0))
             for c in v2.CONCEPTS}
    z = sum(total.values())
    return {c: value / z for c, value in total.items()}


def _form_profile(item: Mapping[str, Any], *, model: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    from cfb_coach.madden.defense_select import call_family

    profile: dict[tuple[str, str], float] = {}
    for ctx in CONTEXT_WEIGHTS:
        for concept in v2.CONCEPTS:
            best: float | None = None
            for play in item["plays"]:
                family = call_family(play)
                if not family:
                    continue
                success, _ = v2.offense_risk(model, ctx, item["formation"], play, family, concept)
                risk = v2.defensive_utility_risk(success, v2.auxiliary_risk(model, ctx, family))
                best = risk if best is None else min(best, risk)
            if best is not None:
                profile[(ctx, concept)] = best
    return profile


def _portfolio_score(chosen: Sequence[Mapping[str, Any]], *, model: Mapping[str, Any],
                     belief: Mapping[str, float],
                     profiles: Mapping[tuple[str, str], dict[tuple[str, str], float]] | None = None,
                     ) -> float:
    """Expected best-of-book coverage across offensive concepts and situations.

    A single do-everything formation only wins if it truly covers the range
    as well as every other catalogued option under the learned model/priors.
    """
    if not chosen:
        return -1.0
    from cfb_coach.madden.defense_select import call_family

    score = 0.0
    for ctx, context_weight in CONTEXT_WEIGHTS.items():
        probs = _scenario_probs(ctx, belief)
        for concept, threat_weight in probs.items():
            eligible = []
            for form in chosen:
                key = (form["source_book"], form["formation"])
                profile = (profiles[key] if profiles is not None else
                           _form_profile(form, model=model))
                if (ctx, concept) in profile:
                    eligible.append(profile[(ctx, concept)])
            if eligible:
                score += context_weight * threat_weight * (1.0 - min(eligible))
    unique_families = {
        call_family(play) for form in chosen for play in form["plays"]
    }
    unique_families.discard(None)
    score += 0.001 * len(unique_families)
    return score


def propose(*, catalogue: Mapping[str, Mapping[str, Sequence[str]]],
            model: Mapping[str, Any], max_formations: int = 15,
            concept_belief: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Model-ranked portfolio over source-confirmed complete stock formations.

    Exact Madden editor availability remains unverified until user confirms it.
    """
    if not 1 <= max_formations <= MAX_FORMATIONS:
        raise ValueError("max_formations must be between 1 and 15")
    if model.get("schema") != v2.SCHEMA:
        raise ValueError("wrong defensive V2 model schema")
    from cfb_coach.madden.defense_select import call_family

    prior = concept_belief or v2.concept_belief(model, [], session="", opponent="", next_seq=1)
    if set(prior) != set(v2.CONCEPTS) or abs(sum(prior.values()) - 1.0) > 1e-6:
        raise ValueError("invalid concept distribution")

    universe = []
    for source in sorted(catalogue):
        for form, source_plays in sorted(catalogue[source].items()):
            plays = list(dict.fromkeys(str(p) for p in source_plays))
            if plays and all(call_family(p) for p in plays):
                universe.append({"source_book": source, "formation": str(form), "plays": plays})
    if not universe:
        raise ValueError("no structurally named defensive formations in catalog")

    # Offline precompute keeps the up-to-15 portfolio optimizer fast even
    # when multiple sources have different play lists for the same formation.
    profiles = {
        (item["source_book"], item["formation"]): _form_profile(item, model=model)
        for item in universe
    }
    portfolio = []
    assigned = set()
    current_score = _portfolio_score(portfolio, model=model, belief=prior, profiles=profiles)
    while len(portfolio) < max_formations:
        choices = [item for item in universe if item["formation"] not in assigned]
        if not choices:
            break
        proposals = []
        for item in choices:
            new_score = _portfolio_score(portfolio + [item], model=model, belief=prior, profiles=profiles)
            proposals.append((new_score - current_score, new_score, item))
        proposals.sort(key=lambda x: (-x[0], x[2]["formation"], x[2]["source_book"]))
        gain, score, best = proposals[0]
        # The first candidate establishes the menu. Subsequent additions must
        # materially improve expected action coverage (> 0.001) or stop.
        if portfolio and gain <= 0.001:
            break
        portfolio.append(best)
        assigned.add(best["formation"])
        current_score = score

    forms = {f["formation"]: f["plays"] for f in portfolio}
    sources = {f["formation"]: f["source_book"] for f in portfolio}
    if not forms:
        raise ValueError("model selected no defensive formations")
    families = Counter(
        call_family(p) for form in portfolio for p in form["plays"]
    )
    result = {
        "schema": "madden.defense.v2.playbook.proposal.v1",
        "model_schema": v2.SCHEMA,
        "n_train": model.get("n_verified", 0),
        "n_train_games": model.get("n_games", 0),
        "n_formations": len(forms),
        "n_plays": sum(map(len, forms.values())),
        "formations": forms,
        "formation_sources": sources,
        "coverage_family_counts": dict(families),
        "model_portfolio_score": round(current_score, 6),
        "max_formations": max_formations,
        "concept_belief": dict(prior),
        "requires_physical_editor_installation": True,
        "installed": False,
        "notes": [
            "Proposed only, not an installed or executable Madden custom book",
            "No fixed formation rotation; model evaluates full source formation play sets",
            "Stock play responsibilities remain unknown where source diagrams are unverified",
            "Portfolio optimizes several offensive threat concepts per situation; no fixed formation quota",
            "Low-evidence personnel and matchup priors are hypotheses, not demonstrated win-rate effects",
            "Candidate universe is the catalogued stock formations only; completeness not assumed",
        ],
    }
    result["proposal_fingerprint"] = hashlib.sha256(
        json.dumps({"forms": forms, "sources": sources}, sort_keys=True).encode()
    ).hexdigest()
    return result
