"""How an offensive concept addresses the defensive picture on this snap.

The trained play score stays primary. This layer adds a capped, sourced
prior only when the current diagnosis supports it. It does not repeat the
down-and-distance or pressure bonuses already applied in football_situation.
"""
from __future__ import annotations

from typing import Any, Mapping

from cfb_coach.madden.model.defensive_diagnosis import diagnose_defense
from cfb_coach.madden.model.football_knowledge import profile_for_play

MATCHUP_VERSION = "concept_matchup.v1"
# Smaller than the 0.06 research cap and far smaller than selection priors.
KNOWLEDGE_CAP = 0.025
# Exits the 0.012 joint indifference band on an otherwise equal play.
_LIVE_BONUS = 0.02
# Stays inside the indifference band. A thin tendency does not decide the snap.
_INFERRED_BONUS = 0.008
_QUICK_REACH_YARDS = 6
_QUICK = frozenset({
    "mesh", "stick", "slant_flat", "curl_flat", "texas_angle", "spacing",
})


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def evaluate_concept_matchup(
    play: str,
    sit: Any = None,
    diagnosis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Context prior for one play. Unknown routes stay unknown."""
    profile = profile_for_play(play)
    principle = profile.get("principle") or {}
    concept_id = profile.get("concept_id")
    if diagnosis is None:
        diagnosis = diagnose_defense(sit)
    observed = diagnosis.get("observed") or {}
    inferred = diagnosis.get("inferred") or {}
    state = diagnosis.get("state")
    shell = observed.get("shell") if state == "observed" else None
    principles = set(observed.get("principles") or []) if state == "observed" else set()
    pressure = bool(observed.get("pressure")) if state == "observed" else False
    down = _int(getattr(sit, "down", None)) if sit is not None else None
    distance = _int(getattr(sit, "distance", None)) if sit is not None else None
    reasons: list[str] = []
    withheld: list[str] = []
    offers: list[float] = []

    def _too_short_to_convert() -> bool:
        return (
            concept_id in _QUICK
            and down in (3, 4)
            and distance is not None
            and distance > _QUICK_REACH_YARDS
        )

    if not concept_id:
        reasons.append("no concept assigned; missing route details stay unknown")
    elif state == "observed" and shell is None and observed.get("structure") == "two_high":
        withheld.append("two-high structure is not Cover 2 or Quarters")
    elif state == "observed":
        man = "man" in principles or shell == "cover_1"
        exploits = set(principle.get("may_exploit") or [])
        if man and concept_id in _QUICK and "man" in exploits:
            if _too_short_to_convert():
                withheld.append(
                    f"{concept_id} can stress man coverage, but a quick concept "
                    f"does not by itself convert {distance} yards"
                )
            elif pressure and principle.get("time_to_develop") == "slow":
                withheld.append("pressure is already handled by the coordinator prior")
            else:
                offers.append(_LIVE_BONUS)
                reasons.append(
                    f"live man coverage; {concept_id} is a sourced underneath answer"
                )
        elif shell and shell in exploits:
            if pressure and principle.get("time_to_develop") == "slow":
                withheld.append(
                    "slow concept against observed pressure; no extra shell bonus"
                )
            elif _too_short_to_convert():
                withheld.append(
                    f"{concept_id} does not convert {distance} yards on its own"
                )
            else:
                offers.append(_LIVE_BONUS)
                reasons.append(
                    f"observed {shell} is a listed stress for {concept_id}; "
                    "principle only, not a Madden diagram"
                )
        elif man or shell:
            withheld.append(
                f"{concept_id} has no sourced answer to the observed look"
            )
    else:
        inferred_shell = inferred.get("shell")
        exploits = set(principle.get("may_exploit") or [])
        confidence = float(inferred.get("confidence") or 0.0)
        if (
            inferred_shell
            and inferred_shell in exploits
            and confidence >= 0.40
            and not _too_short_to_convert()
        ):
            offers.append(_INFERRED_BONUS)
            reasons.append(
                f"inferred {inferred_shell} after shrinkage "
                f"({confidence:.2f}); not this snap's coverage"
            )
        elif diagnosis.get("previous_snap_not_copied"):
            withheld.append("previous coverage was not copied onto this snap")

    if pressure and concept_id in ("screen", "slant_flat", "stick"):
        withheld.append(
            "observed pressure already adjusts quick plays in the coordinator prior"
        )
    if getattr(sit, "red_zone", False) or getattr(sit, "goal_line", False):
        withheld.append("field compression is already in the coordinator prior")

    delta = min(KNOWLEDGE_CAP, max(offers) if offers else 0.0)
    if not reasons:
        reasons.append("no football-knowledge prior on this snap")
    return {
        "version": MATCHUP_VERSION,
        "play": play,
        "concept_id": concept_id,
        "display_name": (principle or {}).get("display_name"),
        "delta": round(delta, 5),
        "confidence": float(profile.get("confidence") or 0.0),
        "layer": profile.get("layer"),
        "reasons": reasons,
        "withheld": withheld,
        "route_diagram": None,
        "player_assignments": None,
        "controller_inputs": None,
        "cap": KNOWLEDGE_CAP,
        "double_count_avoided": [
            "selection_priors",
            "coordinator_pressure",
            "coordinator_field_compression",
            "coordinator_clock",
        ],
        "name_is_not_route_proof": True,
    }
