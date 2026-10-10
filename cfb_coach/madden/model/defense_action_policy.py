"""Bounded defensive play + adjustment selection from verified game controls.

No adjustment is always an explicit competitor. Every non-default plan must
use an existing research source, confirmed controller sequence or a physically
verified/armed custom macro. Unknown last-snap cues never masquerade as live.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from cfb_coach.madden import research_db
from cfb_coach.madden.model.defense_intelligence import tendency
from cfb_coach.madden.situation import concept_family

VERSION = "defense_joint_actions.v1"
MAX_ADJ_BONUS = 0.046
MAX_MACRO_BONUS = 0.052


def _live_concept(sit: Any) -> str | None:
    if getattr(sit, "concept_source", None) != "live":
        return None
    return concept_family(getattr(sit, "concept_hint", None))


def _eligible_tell(
    sit: Any, opponent_id: str, tendencies: Mapping[str, Any] | None
) -> tuple[str | None, str, float]:
    observed = _live_concept(sit)
    if observed:
        return observed, "observed_live", 1.0
    trend = tendency(tendencies, opponent_id, _context(sit))
    # A very strong, multi-snap trend is a soft prior only, not a current tell.
    if trend["ready"] and trend["n"] >= 8 and trend["share"] >= 0.58:
        return str(trend["family"]), "historical_prior", float(trend["confidence"])
    if (getattr(sit, "extras", None) or {}).get("qb_mobile") is True:
        return "scram", "observed_qb_mobility", 0.7
    return None, "unknown", 0.0


def _context(sit: Any) -> str:
    from cfb_coach.madden.defense_select import mix_key
    return mix_key(sit)


def _control_verified(name: str) -> tuple[bool, str]:
    item = research_db.control("defense", name)
    raw = str(item.get("buttons") or "")
    return bool(raw and item.get("confidence") == "confirmed"
                and "VERIFY" not in raw.upper() and item.get("sources")), raw


def researched_adjustments(
    sit: Any, opponent_id: str, tendencies: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    concept, source, strength = _eligible_tell(sit, opponent_id, tendencies)
    if not concept:
        return []
    known_sources = research_db.sources()
    out: list[dict[str, Any]] = []
    for item in research_db.defense_adjustments():
        if concept not in (item.get("answers") or []):
            continue
        source_ids = list(item.get("sources") or [])
        if not source_ids or not all(s in known_sources for s in source_ids):
            continue
        controls = [str(item.get("control") or "")]
        if item.get("extra_control"):
            controls.append(str(item["extra_control"]))
        checked = [_control_verified(key) for key in controls]
        if not all(ok for ok, _ in checked):
            continue
        if source != "observed_live" and source != "observed_qb_mobility" and concept == "scram":
            continue  # historical QB mobility is not enough for a live gamble
        # Research fit is a small proxy, not a learned measured action benefit.
        bonus = min(MAX_ADJ_BONUS, 0.033 * strength) - 0.010 * len(controls)
        if bonus <= 0:
            continue
        out.append({
            "kind": "adjustment", "id": str(item["id"]),
            "label": str(item.get("label") or item["id"]),
            "buttons": " ; then ".join(button for _, button in checked),
            "why": f"{source} {concept}: {item.get('why') or 'researched response'}",
            "research_sources": source_ids, "concept": concept,
            "provenance": source, "influence": round(bonus, 6),
            "executable": True,
        })
    return out


def select_defensive_action(
    sit: Any, db: Any, opponent_id: str,
    formation: str, play: str, *,
    tendencies: Mapping[str, Any] | None = None,
    allowed_macros: bool = True,
) -> dict[str, Any]:
    """Select one legal action (or none) for a chosen installed play."""
    from cfb_coach.madden.model.defense_macro_lab import compatible_verified_macros
    from cfb_coach.madden import playbook
    installed = (playbook.load_books(db).get("defense") or {}).get("formations") or {}
    if play not in installed.get(formation, []):
        return {"kind": "none", "id": "NO_ADJUSTMENT",
                "why": "chosen call is not in confirmed installed book",
                "influence": 0.0, "candidates": []}
    none = {
        "kind": "none", "id": "NO_ADJUSTMENT", "influence": 0.0,
        "why": "run defensive call unchanged", "executable": True,
    }
    choices = [none]
    choices.extend(researched_adjustments(sit, opponent_id, tendencies))
    observed = _live_concept(sit)
    if allowed_macros and observed:
        for macro in compatible_verified_macros(db, opponent_id, formation, play, observed):
            choices.append({
                "kind": "macro", "id": macro["name"], "macro": macro,
                "why": f"physically armed & verified custom adjustment vs observed {observed}",
                "influence": MAX_MACRO_BONUS - 0.018,
                "executable": True, "provenance": "observed_live",
            })
    # Plain unadjusted calls remain preferred in uncertain/tied situations.
    chosen = max(choices, key=lambda x: (float(x.get("influence") or 0.0),
                                         x["kind"] == "none"))
    return {**chosen, "candidates": [
        {k: v for k, v in option.items() if k != "macro"}
        for option in choices
    ], "no_adjustment_compared": True,
        "learned_action_efficacy": False,
        "research_bonus_not_win_probability": True}
