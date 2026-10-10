"""Part 1 defensive knowledge used by live scoring.

The packaged file is ``cfb_coach/data/madden27/defense_knowledge_verified.json``
(schema ``madden27.defense_knowledge.v1``). A tag is loaded only when its
label is ``VERIFIED-M27`` and its confidence is ``high``. Each tag keeps the
``sources`` list from that entry as its provenance.

Medium-confidence and unconfirmed rows live in
``defense_knowledge_hypotheses.json`` with ``live_scoring`` false. This
module does not open that file, and nothing here adds those rows to a score.

Paths are the packaged JSON and the rows passed in by the caller. This
module does not read the process home directory.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "madden27.defense_knowledge.v1"
VERIFIED_LABEL = "VERIFIED-M27"
LIVE_CONFIDENCE = "high"
# Small enough to sit inside the existing near-tie window only when a shell
# named by the verified check list is actually called against all-verticals.
STRESS_DELTA = 0.02
RECENCY_DECAY = 0.85
PROFILE_PRIOR_N = 8.0
OBSERVED_FAMILIES = ("vert", "flood", "cross", "stack", "scram", "run", "rpo")
_CHANGE_MIN = 4

_VERIFIED_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "madden27" / "defense_knowledge_verified.json"
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


@lru_cache(maxsize=1)
def _load_verified() -> dict[str, Any]:
    document = json.loads(_VERIFIED_PATH.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        raise ValueError(f"defense knowledge schema must be {SCHEMA}")
    if document.get("live_scoring") is not True:
        raise ValueError("verified knowledge file is not marked live_scoring")
    items = document.get("items") or []
    if not isinstance(items, list) or not items:
        raise ValueError("verified knowledge file has no tags")
    seen: set[str] = set()
    for item in items:
        label = item.get("label")
        confidence = item.get("confidence")
        sources = item.get("sources")
        tag_id = str(item.get("id") or "")
        if label != VERIFIED_LABEL or confidence != LIVE_CONFIDENCE:
            raise ValueError(
                f"{tag_id or item!r} is not high-confidence {VERIFIED_LABEL}"
            )
        if not tag_id or tag_id in seen:
            raise ValueError("verified knowledge tags need unique ids")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(src, str) and src.strip() for src in sources
        ):
            raise ValueError(f"{tag_id} is missing provenance")
        seen.add(tag_id)
    return document


def verified_tags() -> tuple[dict[str, Any], ...]:
    """High-confidence VERIFIED-M27 tags, provenance copied from each source list."""
    return tuple(dict(item) for item in _load_verified()["items"])


def _tag(tag_id: str) -> dict[str, Any] | None:
    return next((item for item in verified_tags() if item["id"] == tag_id), None)


def _stress_shells() -> tuple[str, ...]:
    """Shells named in the packaged match-check claim, not a second list."""
    item = _tag("match_checks")
    if item is None:
        return ()
    match = re.search(r"\(([^)]+)\)\s*$", str(item.get("claim") or ""))
    if match is None:
        return ()
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def _play_has_shell(play: str, shell: str) -> bool:
    return _normalize(shell) in _normalize(play)


def live_verified_adjustment(sit: Any, play: str) -> dict[str, Any]:
    """Score one installed play with high-confidence VERIFIED-M27 tags only.

    The only numeric rule is the packaged answer ``stress_vs_4verts``: Stress
    is the verified check against all-verticals, and ``match_checks`` names
    the shells that carry those checks (Cover 3 Match, Quarters, Palms,
    Cover 6). Both tags' source lists are copied onto the result. A last-snap
    hint is not a live concept, so it does not score. Other packaged tags
    stay in the catalog; they are settings, controls, or patch notes and do
    not reorder plays.
    """
    from cfb_coach.madden.situation import concept_family

    empty: dict[str, Any] = {
        "delta": 0.0,
        "scored_ids": (),
        "sources": (),
        "why": "No high-confidence VERIFIED-M27 answer changes this play's score.",
        "hypothesis_ids_used": (),
    }
    if getattr(sit, "concept_source", None) != "live":
        return empty
    if concept_family(getattr(sit, "concept_hint", None)) != "vert":
        return empty
    stress = _tag("stress_vs_4verts")
    checks = _tag("match_checks")
    if stress is None or checks is None:
        return empty
    shells = _stress_shells()
    if not shells or not any(_play_has_shell(play, shell) for shell in shells):
        return empty
    sources: list[str] = []
    for item in (stress, checks):
        for src in item["sources"]:
            if src not in sources:
                sources.append(src)
    shell_text = ", ".join(shells)
    return {
        "delta": STRESS_DELTA,
        "scored_ids": (stress["id"], checks["id"]),
        "sources": tuple(sources),
        "why": (
            f"VERIFIED-M27 {stress['id']}: {stress['claim']} "
            f"(sources: {', '.join(stress['sources'])}). "
            f"The play names a shell from {checks['id']}: {shell_text} "
            f"(sources: {', '.join(checks['sources'])})."
        ),
        "hypothesis_ids_used": (),
    }


def explain_call(
    sit: Any, formation: str, play: str, family: str | None, verified: Mapping[str, Any],
) -> str:
    """Why this installed play was scored. Hypothesis tags are not cited."""
    from cfb_coach.madden.defense_select import mix_key

    context = mix_key(sit)
    family_text = family or "unrecognized"
    return (
        f"{context}: {formation} — {play} ({family_text}). "
        f"{verified.get('why') or 'No high-confidence VERIFIED-M27 answer changes this play score.'} "
        "Hypothesis knowledge is not scored."
    )


def formation_packages(chosen: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Situation packages drawn only from formations already in the proposal.

    ``playable`` stays false. Staging a package does not install it.
    """
    def families(row: Mapping[str, Any]) -> set[str]:
        return {str(name) for name in (row.get("families") or ())}

    def name(row: Mapping[str, Any]) -> str:
        return str(row.get("formation") or "").casefold()

    def pack(predicate: Any, purpose: str) -> dict[str, Any]:
        picked: list[str] = []
        for row in chosen:
            formation = str(row.get("formation") or "")
            if formation and predicate(row) and formation not in picked:
                picked.append(formation)
        return {
            "formations": picked,
            "playable": False,
            "status": "staged",
            "purpose": purpose,
            "note": "Proposal only. A formation is not playable until it is confirmed installed.",
        }

    heavy_tokens = ("4-3", "4-4", "3-4", "46", "bear", "goal")
    spread_tokens = ("dime", "nickel", "dollar", "quarter")
    return {
        "early_downs": pack(
            lambda row: True,
            "Early downs can use any formation in this staged proposal.",
        ),
        "passing_downs": pack(
            lambda row: bool(families(row) & {"two_high", "cover2"}),
            "Passing downs prefer staged formations that carry two-high or Cover 2 calls.",
        ),
        "goal_line": pack(
            lambda row: any(token in name(row) for token in heavy_tokens),
            "Goal-line package is the staged heavy formations, not a research-only front.",
        ),
        "two_minute": pack(
            lambda row: any(token in name(row) for token in spread_tokens)
            or "two_high" in families(row),
            "Two-minute package is the staged lighter or two-high formations.",
        ),
        "spread": pack(
            lambda row: any(token in name(row) for token in spread_tokens),
            "Spread package is the staged nickel, dime, dollar, and quarter formations.",
        ),
        "heavy": pack(
            lambda row: any(token in name(row) for token in heavy_tokens),
            "Heavy package is the staged 4-3, 4-4, 3-4, 46, bear, and goal-line formations.",
        ),
        "mobile_qb": pack(
            lambda row: bool(families(row) & {"man", "pressure"}),
            "Mobile-QB package is staged formations that already have a man or pressure call.",
        ),
    }


def _mode(labels: Sequence[str]) -> str | None:
    if not labels:
        return None
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(sorted(counts), key=lambda label: (counts[label], label))


def _profile_for(observations: Sequence[str]) -> dict[str, Any]:
    families = OBSERVED_FAMILIES
    weights = [RECENCY_DECAY ** (len(observations) - 1 - index) for index in range(len(observations))]
    totals = {family: 0.0 for family in families}
    for family, weight in zip(observations, weights):
        if family in totals:
            totals[family] += weight
    weight_sum = sum(totals.values())
    prior_each = PROFILE_PRIOR_N / len(families)
    shares = {
        family: (totals[family] + prior_each) / (weight_sum + PROFILE_PRIOR_N)
        for family in families
    }
    family = max(families, key=lambda name: (shares[name], name))
    older_family = newer_family = None
    change = False
    if len(observations) >= _CHANGE_MIN * 2:
        mid = len(observations) // 2
        older, newer = list(observations[:mid]), list(observations[mid:])
        older_family, newer_family = _mode(older), _mode(newer)
        change = (
            older_family is not None
            and newer_family is not None
            and older_family != newer_family
            and len(older) >= _CHANGE_MIN
            and len(newer) >= _CHANGE_MIN
        )
    return {
        "ready": True,
        "n": len(observations),
        "family": family,
        "share": round(shares[family], 5),
        "uncertainty": round(PROFILE_PRIOR_N / (weight_sum + PROFILE_PRIOR_N), 5),
        "shrinkage_prior_n": PROFILE_PRIOR_N,
        "recency_decay": RECENCY_DECAY,
        "change_detected": change,
        "older_family": older_family,
        "newer_family": newer_family,
        "previous_snap_concept": observations[-1],
        "current_snap_known": False,
        "previous_snap_is_current_play": False,
        "live_score_delta": 0.0,
        "historical_only": True,
    }


def fit_opponent_profiles(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recency-weighted opponent concepts from verified human executions.

    The newest snap is the heaviest historical observation. It is not the
    current play. A change between the older and newer halves is reported
    and does not add a live score. Sample-size shrinkage pulls a short
    history toward the even prior.
    """
    from cfb_coach.madden.model.defense_intelligence import verified_human_rows
    from cfb_coach.madden.situation import concept_family

    grouped: dict[str, list[str]] = {}
    for row in verified_human_rows(rows):
        family = concept_family(str(row.get("concept_seen") or ""))
        if family not in OBSERVED_FAMILIES:
            continue
        opponent = str(row["opponent_id"])
        grouped.setdefault(opponent, []).append(family)
    return {
        "schema": SCHEMA,
        "opponents": {
            opponent: _profile_for(observations)
            for opponent, observations in grouped.items()
        },
        "live_score_delta": 0.0,
        "uses_hypothesis_tags": False,
        "current_snap_known": False,
    }
