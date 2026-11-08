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


def assignment_status(structure: Any) -> tuple[str, float]:
    """("verified" | "name_only" | "unknown", confidence). Verified needs known routes from a documented tier."""
    try:
        n_known = int(getattr(structure, "n_known", 0) or 0)
        tiers = [t for _, _, t in (getattr(structure, "routes", None) or ())]
    except Exception:  # noqa: BLE001
        n_known, tiers = 0, []
    documented = [t for t in tiers if t in DOCUMENTED_TIERS]
    if n_known > 0 and len(documented) == n_known:
        return "verified", min(fpri.TIER_TRUST[t] for t in documented)
    concepts = fpri.play_concepts(structure) if structure is not None else {}
    if concepts:
        return "name_only", fpri.NAME_TRUST
    return "unknown", 0.0


def score_pair(art: Any, *, formation: str, play: str, probe: Any, mix: Mapping[str, float] | None, opponent_id: str,
               priors_on: bool, learned: Mapping | None = None, structure: Any | None = None,
               mix_source: str | None = None) -> PregameScore:
    pred = experimental_model.predict_success(
        art, formation=formation, play=play, down=getattr(probe, "down", None), distance=getattr(probe, "distance", None),
        yardline=getattr(probe, "yardline", None), opponent_id=opponent_id,
        coverage_hint=None, coverage_source="none", heuristic_bonus=0.0)
    base = float(pred["probability"])
    s = structure if structure is not None else sm.structure_any(formation, play)
    status, conf = assignment_status(s)
    clean_mix = {str(k): float(v) for k, v in (mix or {}).items() if float(v) > 0}
    src = "unknown" if not clean_mix else (mix_source or "counterfactual")
    edge, se, cells, mass = 0.0, 0.0, [], 0.0
    if priors_on and clean_mix:
        e = fpri.matchup_edge(s, clean_mix, learned=learned)
        edge, se, cells, mass = float(e["edge"]), float(e["se"]), list(e["cells"]), float(e.get("mass") or 0.0)
    total = sm._sigmoid(sm._logit(base) + edge) if edge else base
    return PregameScore(
        formation=formation, play=play, base_probability=base, matchup_edge=edge, total_probability=total,
        mix=clean_mix, mix_source=src, assignment_status=status, assignment_confidence=conf, cells=cells,
        components={"edge_se": se, "known_mix_mass": mass, "model_uncertainty": pred.get("uncertainty"),
                    "evidence_quality": pred.get("evidence_quality"), "priors_on": bool(priors_on),
                    "reads": "documented routes" if status == "verified" else "UNKNOWN (no documented assignments)"})


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
