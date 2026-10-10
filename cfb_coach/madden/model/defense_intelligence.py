"""Defensive football knowledge and opponent offense tendencies.

Only verified, executed HUMAN user-game defensive snaps can enter the data fit.
Observed post-snap concepts are historical evidence, NEVER knowledge of the
offense's current call. Unknown concepts and coverage remain unknown.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from cfb_coach.madden.defense_select import COUNTERS
from cfb_coach.madden.situation import concept_family

VERSION = "madden.defense.intelligence.v1"
MIN_OPPONENT_N = 6
MIN_CONTEXT_N = 4
PRIOR_N = 8.0
MAX_OPPONENT_INFLUENCE = 0.075
MAX_LIVE_CONCEPT_INFLUENCE = 0.10

# Football-informed hypotheses, NOT verified Madden 27 coverage-match diagrams.
PRINCIPLES = {
    "vert": "Do not leave isolated deep defenders unsupported; protect explosives.",
    "flood": "Defend layered sideline routes with sound curl/flat/deep responsibilities.",
    "cross": "Communicate crossers and avoid uncovered shallow windows.",
    "stack": "Anticipate traffic and free releases versus bunch/stack spacing.",
    "run": "Maintain gap integrity and run fits; do not abandon the box blindly.",
    "rpo": "Respect both run and immediate perimeter-pass threats.",
    "scram": "Retain edge/contain responsibility against a mobile quarterback.",
}
STRATEGIC_CONTEXTS = (
    "early", "short", "medium", "long", "red_zone", "goal_line", "two_minute",
)


def context_key(row: Mapping[str, Any]) -> str:
    from cfb_coach.madden.model.defense_coordinator import _bucket
    return _bucket(
        row.get("down"), row.get("distance"),
        red_zone=bool(row.get("red_zone")), goal_line=bool(row.get("goal_line")),
        two_minute=bool(row.get("two_minute")),
    )


def verified_human_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Never turn recommended actions, CPU data, or unverified logs into training."""
    return [
        dict(r) for r in rows
        if str(r.get("opponent_type") or "").lower() == "human"
        and bool(str(r.get("opponent_id") or "").strip())
        and bool(str(r.get("game_id") or "").strip())
        and r.get("supervised_eligible") is True
        and str(r.get("executed_verification") or "") == "verified"
        and str(r.get("executed_status") or "") == "identified"
        and r.get("executed_play")
        and r.get("label_available")
    ]


def fit_opponent_offense(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Post-snap observed concepts -> small probabilistic *future* tendency priors.

    A concept_seen is an observation label, not a visible pre-snap play.
    Missing concept_seen rows are excluded from concept denominators, never
    guessed from the defensive play we selected.
    """
    buckets: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    games: dict[str, set[str]] = defaultdict(set)
    observed = 0
    for row in verified_human_rows(rows):
        raw = row.get("concept_seen")
        family = concept_family(str(raw)) if raw else None
        if family not in PRINCIPLES:
            continue
        opponent = str(row["opponent_id"])
        context = context_key(row)
        games[opponent].add(str(row["game_id"]))
        observed += 1
        for scope in ("all", context):
            counts = buckets[opponent][scope]
            counts[family] = counts.get(family, 0) + 1
    return {
        "schema": VERSION, "observed_post_snap_concepts": observed,
        "opponents": {
            opponent: {
                "contexts": dict(scopes),
                "games": len(games[opponent]),
                "sample_count": sum(scopes.get("all", {}).values()),
            } for opponent, scopes in buckets.items()
        },
        "source": "user-reported post-snap concept_seen on verified defensive executions",
        "current_snap_knowledge": False, "human_games_only": True,
    }


def tendency(
    artifact: Mapping[str, Any] | None, opponent_id: str, context: str,
) -> dict[str, Any]:
    scopes = ((artifact or {}).get("opponents") or {}).get(opponent_id) or {}
    counts = (scopes.get("contexts") or {}).get(context) or {}
    n = sum(int(v) for v in counts.values())
    required = MIN_CONTEXT_N
    if n < required:
        counts = (scopes.get("contexts") or {}).get("all") or {}
        n = sum(int(v) for v in counts.values())
        required = MIN_OPPONENT_N
    if n < required:
        return {"ready": False, "n": n, "family": None, "confidence": 0.0}
    fam = max(counts, key=counts.get)
    top = int(counts[fam])
    # Shrunk multinomial against an unknown eight-bucket football prior.
    share = (top + PRIOR_N / len(PRINCIPLES)) / (n + PRIOR_N)
    confidence = (n / (n + PRIOR_N)) * share
    return {
        "ready": True, "n": n, "family": fam,
        "share": round(share, 5), "confidence": round(confidence, 5),
        "historical_only": True,
        "note": "This is a past tendency, NOT a live observed offensive play.",
    }


def score_defensive_knowledge(
    *,
    family: str | None, sit: Any, opponent_id: str,
    tendencies: Mapping[str, Any] | None = None,
    risks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from cfb_coach.madden.defense_select import mix_key
    ctx = mix_key(sit)
    observed = None
    raw = getattr(sit, "concept_hint", None)
    if raw and getattr(sit, "concept_source", None) == "live":
        observed = concept_family(str(raw))
    opponent = tendency(tendencies, opponent_id, ctx)
    live_delta = 0.0
    learned_delta = 0.0
    if observed in PRINCIPLES and family:
        live_delta = max(
            -MAX_LIVE_CONCEPT_INFLUENCE,
            min(MAX_LIVE_CONCEPT_INFLUENCE, 0.17 * COUNTERS.get(observed, {}).get(family, 0.0)),
        )
    if opponent["ready"] and family:
        concept = opponent["family"]
        learned_delta = max(
            -MAX_OPPONENT_INFLUENCE,
            min(MAX_OPPONENT_INFLUENCE,
                0.19 * COUNTERS.get(concept, {}).get(family, 0.0) * opponent["confidence"]),
        )
    qb_mobile = (getattr(sit, "extras", None) or {}).get("qb_mobile")
    mobile_adjustment = 0.0
    if qb_mobile is True and family == "pressure":
        mobile_adjustment = -0.025  # pressure without contain may lose the edge
    risk = defensive_risk_delta(risks, opponent_id, ctx, family)
    return {
        "delta": round(live_delta + learned_delta + mobile_adjustment + risk["delta"], 6),
        "opponent_outcome_risk": risk,
        "live_concept": observed,
        "opponent_tendency": opponent,
        "principle": PRINCIPLES.get(observed or opponent.get("family"), "Maintain assignment integrity."),
        "evidence": {
            "live": observed is not None,
            "historical_tendency": bool(opponent["ready"]),
            "counterfactual_success_claim": False,
            "exact_route_assignments_verified": False,
        },
    }



def fit_opponent_outcome_risks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Opponent-level outcomes need NOT claim which defensive play was executed.

    These are actual historical snap RESULTS (yards/downs) separated from the
    possibly unverified defensive call. Recommendations contribute no actions.
    Unknown yards or unknown opponent identity are excluded.
    """
    contexts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "explosives": 0, "conversions": 0,
                                     "conversion_opportunities": 0})
    )
    game_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if str(row.get("opponent_type") or "").lower() != "human":
            continue
        if str(row.get("side") or "") != "defense":
            continue
        gid, opponent = str(row.get("game_id") or ""), str(row.get("opponent_id") or "")
        if not gid or not opponent or not row.get("label_available"):
            continue
        try:
            yards = int(row["yards"])
        except (KeyError, ValueError, TypeError):
            continue
        context = context_key(row)
        game_ids[opponent].add(gid)
        for scope in ("all", context):
            stats = contexts[opponent][scope]
            stats["n"] += 1
            stats["explosives"] += int(yards >= 20)
            try:
                distance = int(row["distance"])
            except (KeyError, ValueError, TypeError):
                distance = None
            if distance is not None and distance > 0:
                stats["conversion_opportunities"] += 1
                stats["conversions"] += int(yards >= distance)
    return {
        "schema": VERSION,
        "basis": "historical_opponent_outcome_only",
        "opponents": {
            opponent: {"games": len(game_ids[opponent]), "contexts": dict(scopes)}
            for opponent, scopes in contexts.items()
        },
        "observed_outcomes": sum(
            scopes["all"]["n"] for scopes in contexts.values()
        ),
        "defensive_play_attribution": False,
        "requires_verified_own_execution": False,
        "live_effect_bound": 0.025,
    }


def defensive_risk_delta(
    artifact: Mapping[str, Any] | None, opponent_id: str, context: str,
    coverage_family: str | None,
) -> dict[str, Any]:
    """Tiny opponent risk prior; no particular coverage credited for stops."""
    opp = ((artifact or {}).get("opponents") or {}).get(opponent_id) or {}
    contexts = opp.get("contexts") or {}
    data = contexts.get(context) or contexts.get("all") or {}
    n = int(data.get("n") or 0)
    if n < 8:
        return {"delta": 0.0, "n": n, "ready": False}
    rate = (int(data.get("explosives") or 0) + 2.0) / (n + 12.0)
    uncertainty = n / (n + 12.0)
    # High observed opponent explosives suggest protecting deep space.
    # This is a bounded coaching hypothesis, NOT proof about any coverage.
    risk = max(0.0, rate - 0.12) * uncertainty
    delta = 0.0
    if coverage_family in ("two_high", "cover2"):
        delta = min(0.025, 0.15 * risk)
    elif coverage_family == "pressure":
        delta = -min(0.025, 0.15 * risk)
    return {
        "delta": round(delta, 6), "n": n, "ready": True,
        "explosive_rate_shrunk": round(rate, 5),
        "no_play_success_attribution": True,
        "historical_only": True,
    }
