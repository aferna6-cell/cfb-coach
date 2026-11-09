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


def _portfolio_score(chosen: Sequence[Mapping[str, Any]], *, model: Mapping[str, Any],
                     belief: Mapping[str, float]) -> float:
    if not chosen:
        return -1.0
    # Optimize the quality of the best legal coverage we can access under
    # each context, not an arbitrary equal quota for coverage families.
    from cfb_coach.madden.defense_select import call_family
    score = 0.0
    for ctx, weight in CONTEXT_WEIGHTS.items():
        best = None
        for form in chosen:
            for play in form["plays"]:
                fam = call_family(play)
                if fam is None:
                    continue
                risk = _risk(model, ctx, form["formation"], play, fam, belief)
                best = risk if best is None else min(best, risk)
        if best is not None:
            score += weight * (1.0 - best)
    # Very modest action breadth preference (not a mandatory fixed ratio).
    unique_families = {call_family(play) for form in chosen for play in form["plays"]}
    unique_families.discard(None)
    score += 0.002 * len(unique_families)
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

    portfolio = []
    assigned = set()
    current_score = _portfolio_score(portfolio, model=model, belief=prior)
    while len(portfolio) < max_formations:
        choices = [item for item in universe if item["formation"] not in assigned]
        if not choices:
            break
        proposals = []
        for item in choices:
            new_score = _portfolio_score(portfolio + [item], model=model, belief=prior)
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
            "Candidate universe is the catalogued stock formations only; completeness not assumed",
        ],
    }
    result["proposal_fingerprint"] = hashlib.sha256(
        json.dumps({"forms": forms, "sources": sources}, sort_keys=True).encode()
    ).hexdigest()
    return result
