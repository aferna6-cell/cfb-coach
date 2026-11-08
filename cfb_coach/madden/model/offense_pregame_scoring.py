"""Pregame (formation, play) scoring for the offensive designer (#130).

The trained ML model owns expected play value (``experimental_model.predict_success`` on a generic probe, never with a
live/last coverage source: a hypothetical pregame shell was not observed). Football knowledge enters ONLY as the same
``football_priors.matchup_edge`` logit lift the live ranker uses, under an explicit coverage MIXTURE and an explicit
``priors_on`` flag (the env var / meta switch is never read or written here). Empty mix => edge 0 and honest uncertainty.
Assignment status is reported honestly: name hints are never promoted to verified routes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model import football_priors as fpri
from cfb_coach.madden.model import offense_structural_model as sm

DOCUMENTED_TIERS = frozenset({"in_game_verified", "diagram_verified", "source_documented"})
MIX_SOURCES = ("counterfactual", "opponent_history", "unknown")
ASSIGNMENT_STATUSES = ("verified", "high_confidence_extracted", "partial", "name_only", "unknown")
# Provisional (Ace-derived) geometry is never documented; reuse the documented inference trust, no new magic number.
PROVISIONAL_TRUST = fpri.TIER_TRUST["football_inference"]
PROVISIONAL_ROUTE_TIERS = frozenset({"high_confidence_extracted", "provisional", "ace_extracted"})
PROVISIONAL_READS = "UNKNOWN (provisional structure; not verified)"
UNVERIFIED_READS = "UNKNOWN (no documented assignments)"
_FEATURE_TIER_ALIASES = {"verified_assignments": "verified", "verified": "verified",
                         "high_confidence_extracted": "high_confidence_extracted", "partial": "partial",
                         "name_only": "name_only", "unknown": "unknown"}


@dataclass(frozen=True)
class PregameScore:
    formation: str
    play: str
    base_probability: float
    matchup_edge: float
    total_probability: float
    mix: dict[str, float]
    mix_source: str
    assignment_status: str
    assignment_confidence: float
    cells: list = field(default_factory=list)
    components: dict = field(default_factory=dict)
    # #132 additions (defaults keep positional/keyword callers working).
    shrunk_probability: float | None = None
    score_calibrated: bool = False
    score_available: bool = True
    global_rate: float | None = None


def _route_tiers(structure: Any) -> tuple[int, list[str]]:
    try:
        n_known = int(getattr(structure, "n_known", 0) or 0)
        tiers = [t for _, _, t in (getattr(structure, "routes", None) or ())]
    except Exception:  # noqa: BLE001
        n_known, tiers = 0, []
    return n_known, tiers


def _is_provisional(structure: Any, tiers: list[str]) -> bool:
    if structure is None:
        return False
    if bool(getattr(structure, "ace_image_derived", False)):
        return True
    if str(getattr(structure, "feature_tier", "") or "") == "high_confidence_extracted":
        return True
    return any(t in PROVISIONAL_ROUTE_TIERS for t in tiers)


def _name_or_unknown(structure: Any) -> tuple[str, float]:
    concepts = fpri.play_concepts(structure) if structure is not None else {}
    if concepts:
        return "name_only", fpri.NAME_TRUST
    return "unknown", 0.0


def assignment_status(structure: Any, *, feature_tier: str | None = None,
                      model_use_permitted: bool | None = True) -> tuple[str, float]:
    """(status, confidence); status in ASSIGNMENT_STATUSES.

    ``verified`` ALWAYS requires known routes that are all from a documented tier, whatever a feed claims.
    ``partial``: some (not all) known routes documented; confidence = documented share * min documented trust.
    ``high_confidence_extracted``: provisional Ace geometry with model use permitted; PROVISIONAL_TRUST.
    All-``football_inference`` first-party routes are not promoted: they fall through to name_only/unknown."""
    n_known, tiers = _route_tiers(structure)
    documented = [t for t in tiers if t in DOCUMENTED_TIERS]
    if n_known > 0 and len(documented) == n_known:
        return "verified", min(fpri.TIER_TRUST[t] for t in documented)
    if n_known > 0 and documented:
        return "partial", (len(documented) / n_known) * min(fpri.TIER_TRUST[t] for t in documented)
    tier = feature_tier if feature_tier is not None else getattr(structure, "feature_tier", None)
    claimed = _FEATURE_TIER_ALIASES.get(str(tier or ""), None)
    permitted = model_use_permitted is not False and getattr(structure, "model_use_permitted", True) is not False
    if permitted and (claimed == "high_confidence_extracted" or _is_provisional(structure, tiers)):
        return "high_confidence_extracted", PROVISIONAL_TRUST
    status, conf = _name_or_unknown(structure)
    if permitted and claimed == "partial":
        # Feed-claimed partial geometry without documented routes: keep alternatives, no more trust than a name hint.
        return "partial", fpri.NAME_TRUST
    if claimed == "unknown":
        return "unknown", 0.0
    # A claimed verified/partial tier without documented routes is never trusted beyond the name hint.
    return status, conf


def resolve_global_rate(art: Any) -> float:
    """Documented global rate: nested empirical ``art.base`` first, then ``art.global_rate``, else 0.5."""
    base = getattr(art, "base", None)
    for cand in ((base.get("global_rate") if isinstance(base, Mapping) else getattr(base, "global_rate", None))
                 if base is not None else None, getattr(art, "global_rate", None)):
        try:
            if cand is not None:
                return float(cand)
        except (TypeError, ValueError):
            continue
    return 0.5


def score_pair(art: Any, *, formation: str, play: str, probe: Any, mix: Mapping[str, float] | None, opponent_id: str,
               priors_on: bool, learned: Mapping | None = None, structure: Any | None = None,
               mix_source: str | None = None, feature_tier: str | None = None,
               model_use_permitted: bool | None = True) -> PregameScore:
    pred = experimental_model.predict_success(
        art, formation=formation, play=play, down=getattr(probe, "down", None), distance=getattr(probe, "distance", None),
        yardline=getattr(probe, "yardline", None), opponent_id=opponent_id,
        coverage_hint=None, coverage_source="none", heuristic_bonus=0.0)
    base = float(pred["probability"])
    s = structure if structure is not None else sm.structure_any(formation, play)
    status, conf = assignment_status(s, feature_tier=feature_tier, model_use_permitted=model_use_permitted)
    clean_mix = {str(k): float(v) for k, v in (mix or {}).items() if float(v) > 0}
    src = "unknown" if not clean_mix else (mix_source or "counterfactual")
    edge, se, cells, mass = 0.0, 0.0, [], 0.0
    if priors_on and clean_mix:
        e = fpri.matchup_edge(s, clean_mix, learned=learned)
        edge, se, cells, mass = float(e["edge"]), float(e["se"]), list(e["cells"]), float(e.get("mass") or 0.0)
    raw_edge = edge
    provisional = status == "high_confidence_extracted" or (status == "partial" and not _route_tiers(s)[0])
    if provisional and edge:
        edge = edge * conf  # confidence-weighted prior: shrink toward 0 by (1 - confidence)
    total = sm._sigmoid(sm._logit(base) + edge) if edge else base
    # Cold start: a prior-driven / constant 0.5 base is not calibrated certainty.
    eq = pred.get("evidence_quality") or getattr(art, "evidence_quality", None)
    calibrated = not (eq == "prior_driven" or abs(base - 0.5) < 1e-9)
    g = resolve_global_rate(art)
    if calibrated:
        shrunk = total
    else:
        # Shrink the uncalibrated base toward the documented global rate, then add the already-trusted edge.
        sb = min(max(g + conf * (base - g), 1e-6), 1 - 1e-6)
        shrunk = sm._sigmoid(sm._logit(sb) + edge) if edge else sb
    if status == "verified":
        reads = "documented routes"
    elif provisional:
        reads = PROVISIONAL_READS
    else:
        reads = UNVERIFIED_READS
    return PregameScore(
        formation=formation, play=play, base_probability=base, matchup_edge=edge, total_probability=total,
        mix=clean_mix, mix_source=src, assignment_status=status, assignment_confidence=conf, cells=cells,
        components={"edge_se": se, "known_mix_mass": mass, "model_uncertainty": pred.get("uncertainty"),
                    "evidence_quality": pred.get("evidence_quality"), "priors_on": bool(priors_on),
                    "reads": reads, "raw": total, "shrunk": shrunk, "raw_matchup_edge": raw_edge,
                    "global_rate": g, "score_calibrated": calibrated,
                    "shrinkage": "none (calibrated)" if calibrated else "base toward global_rate by confidence"},
        shrunk_probability=shrunk, score_calibrated=calibrated, score_available=True, global_rate=g)


def coverage_mix_from_seen(raw_labels: Iterable[str | None]) -> dict[str, float]:
    """Raw coverage_seen labels -> normalized mix. Only SPECIFIC labels map to variants; coarse labels map to unspecified
    keys; anything else is dropped as unknown (never guessed)."""
    counts: dict[str, float] = {}
    for raw in raw_labels:
        t = re.sub(r"[^a-z0-9 ]+", " ", str(raw or "").lower()).strip()
        if not t:
            continue
        key = None
        if "palm" in t:
            key = "cover_4_palms"
        elif "quarter" in t:
            key = "cover_4_quarters"
        elif "robber" in t:
            key = "cover_1_robber"
        elif "trap" in t:
            key = "trap_2"
        elif re.search(r"\b(cover|cov|c) ?3\b", t) and "match" in t:
            key = "cover_3_match"
        elif re.search(r"\b(cover|cov|c) ?[23]\b", t) and "spot" in t:
            key = "cover_2_spot" if re.search(r"\b(cover|cov|c) ?2\b", t) else "cover_3_spot"
        elif re.search(r"\b(cover|cov|c) ?0\b", t):
            key = "cover_0"
        elif re.search(r"\b(cover|cov|c) ?1\b", t) and ("blitz" in t or "pressure" in t):
            key = "cover_1_pressure"
        elif re.search(r"\b(cover|cov|c) ?2\b", t) and "man" in t:
            key = "cover_2_man"
        elif re.search(r"\b(cover|cov|c) ?1\b", t) or t == "man":
            key = "cover_1_unspecified"
        elif re.search(r"\b(cover|cov|c) ?2\b", t):
            key = "cover_2_unspecified"
        elif re.search(r"\b(cover|cov|c) ?3\b", t):
            key = "cover_3_unspecified"
        elif re.search(r"\b(cover|cov|c) ?4\b", t):
            key = "cover_4_unspecified"
        if key:
            counts[key] = counts.get(key, 0.0) + 1.0
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()} if tot else {}
