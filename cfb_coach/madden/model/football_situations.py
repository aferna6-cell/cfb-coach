"""Structured football situation intelligence for the Madden offensive coordinator.

This module turns a parsed :class:`~cfb_coach.situation.Situation` into an
explicit, auditable assessment that separates three kinds of knowledge:

* OBSERVED facts   — down/distance, field zone, score, an explicitly live
  pre-snap defensive look.
* INFERRED signals — last-snap coverage, accumulated in-game tendencies with
  their sample sizes. These are priors, never proof of the present coverage.
* UNKNOWN          — everything else. Unknown stays unknown; no fabricated
  certainty is ever written into the assessment.

Every strategic prior is a named, testable entry in :data:`SITUATION_PRIORS`
with its magnitude and rationale, instead of hidden bonuses sprinkled through
selection code. Magnitudes are situation-aware proxies for conversion /
possession value, clearly labeled as football priors — they are NOT win
probability or expected points estimates, which the verified data cannot
support yet.
"""
from __future__ import annotations

from typing import Any, Mapping

from cfb_coach.madden.catalog import is_deep, is_run, zone_fit
from cfb_coach.madden.model.experimental_model import _play_concept, _play_family

ASSESSMENT_SCHEMA = "madden.football_situation.v1"

# Named, testable strategic priors. ``delta`` values are selection-score
# proxies on the same scale as model probabilities (±0.0..0.5); the basis
# field says what grounds each one. Keep deltas aligned with the historical
# `_situational_adjustment` magnitudes so existing verified behavior and
# regression tests remain meaningful.
SITUATION_PRIORS: dict[str, dict[str, Any]] = {
    "long_down_run": {
        "delta": -0.45, "basis": "football_prior",
        "rationale": "A standard run rarely reaches a 7+ yard conversion marker on 3rd/4th down",
    },
    "long_down_screen": {
        "delta": -0.09, "basis": "football_prior",
        "rationale": "Screens start behind the line and need broken tackles to reach a long marker",
    },
    "long_down_route_past_sticks": {
        "delta": 0.025, "basis": "football_prior",
        "rationale": "Deep-breaking routes can realistically reach the first-down marker",
    },
    "short_yardage_run": {
        "delta": 0.025, "basis": "football_prior",
        "rationale": "Short-yardage conversions favor efficient interior runs",
    },
    "short_yardage_slow_shot": {
        "delta": -0.035, "basis": "football_prior",
        "rationale": "Long-developing shots risk an efficient 1-2 yard conversion",
    },
    "second_short_upside": {
        "delta": 0.02, "basis": "football_prior",
        "rationale": "2nd-and-short is a low-cost down to take a higher-upside shot",
    },
    "goal_line_compressed_deep": {
        "delta": -0.05, "basis": "football_prior",
        "rationale": "Compressed goal-line spacing removes vertical route room",
    },
    "goal_line_run_or_quick": {
        "delta": 0.03, "basis": "football_prior",
        "rationale": "Runs and quick one-cut throws fit compressed goal-line spacing",
    },
    "backed_up_risk": {
        "delta": -0.025, "basis": "football_prior",
        "rationale": "Deep drops and screens near our goal line risk safeties and short fields",
    },
    "two_minute_trailing_run": {
        "delta": -0.07, "basis": "football_prior",
        "rationale": "Trailing in the two-minute drill, inbounds runs burn irreplaceable clock",
    },
    "two_minute_leading_run": {
        "delta": 0.04, "basis": "football_prior",
        "rationale": "Leading late, a run keeps the clock moving and protects possession",
    },
    "protect_lead_possession": {
        "delta": 0.03, "basis": "football_prior",
        "rationale": "With a late lead, possession security and clock consumption carry extra value",
    },
    "protect_lead_low_percentage_shot": {
        "delta": -0.03, "basis": "football_prior",
        "rationale": "A low-percentage deep shot stops the clock and risks the lead",
    },
    "trailing_late_clock_cost": {
        "delta": -0.03, "basis": "football_prior",
        "rationale": "Trailing late, slow-developing clock-consuming calls cost comeback time",
    },
}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _score_posture(sit: Any) -> dict[str, Any]:
    """Protect / trail / neutral / prevent, or unknown when no score is set."""
    try:
        from cfb_coach.game_score import classify

        ctx = classify(sit)
    except Exception:  # noqa: BLE001 — score context is optional
        ctx = None
    if ctx is None:
        return {"posture": "unknown", "margin": None, "late": None, "strength": 0.0}
    return {
        "posture": ctx.phase, "margin": ctx.margin,
        "late": ctx.late, "strength": ctx.strength,
    }


def _defensive_knowledge(
    sit: Any, *, db: Any = None, opponent_id: str = "",
) -> dict[str, Any]:
    """Split what we know about the defense into observed / inferred / unknown."""
    from cfb_coach.madden.playcaller import coverage_class

    cov = getattr(sit, "coverage_hint", None)
    source = getattr(sit, "coverage_source", None) or "none"
    cls = coverage_class(cov) if cov else None
    observed = cov if source == "live" else None
    inferred = cov if source == "last" else None
    repeated_count = 0
    if db is not None and opponent_id and cov:
        try:
            from cfb_coach.tendency import coverage_count

            repeated_count = int(coverage_count(db, opponent_id, cov, sit))
        except Exception:  # noqa: BLE001 — tendency store is optional
            repeated_count = 0
    if observed:
        credibility = "observed_live"
    elif inferred and repeated_count >= 2:
        credibility = "inferred_repeated"
    elif inferred:
        credibility = "inferred_single"
    else:
        credibility = "unknown"
    return {
        "observed_look": observed,
        "observed_class": cls if observed else None,
        "inferred_tendency": inferred,
        "inferred_class": cls if inferred else None,
        "tendency_sample": repeated_count,
        "credibility": credibility,
        "note": (
            "A previous coverage observation is evidence about tendencies, "
            "never proof of the present coverage."
        ),
    }


def assess_situation(
    sit: Any, *, db: Any = None, opponent_id: str = "",
) -> dict[str, Any]:
    """Structured football assessment of one pre-snap situation.

    Observation-only: nothing here reads this snap's outcome, and defensive
    knowledge is explicitly labeled by credibility.
    """
    down = _int_or_none(getattr(sit, "down", None))
    distance = _int_or_none(getattr(sit, "distance", None))
    goal_line = bool(getattr(sit, "goal_line", False))
    red_zone = bool(getattr(sit, "red_zone", False)) or goal_line
    two_minute = bool(getattr(sit, "two_minute", False))
    yardline = _int_or_none(getattr(sit, "yardline", None))
    quarter = _int_or_none((getattr(sit, "extras", None) or {}).get("quarter"))
    score = _score_posture(sit)

    phases: list[str] = []
    if goal_line:
        phases.append("goal_line")
    elif red_zone:
        phases.append("red_zone")
    short = down is not None and distance is not None and down in (3, 4) and 0 < distance <= 2
    long_down = down is not None and distance is not None and down in (3, 4) and distance >= 7
    if short:
        phases.append("short_yardage")
    if long_down:
        phases.append("third_or_fourth_long")
    if down == 2 and distance is not None and 0 < distance <= 2:
        phases.append("second_and_short")
    backed_up = yardline is not None and yardline <= 10 and not red_zone
    if backed_up:
        phases.append("backed_up")
    if two_minute:
        phases.append("two_minute")
    if score["posture"] in ("protect", "prevent") and score["late"]:
        phases.append("protect_lead_late")
    if score["posture"] == "trail" and score["late"]:
        phases.append("trailing_late")
    if not phases:
        phases.append("normal")

    if two_minute and score["posture"] == "trail":
        clock = "preserve_clock"
    elif score["posture"] in ("protect", "prevent") and (two_minute or score["late"]):
        clock = "consume_clock"
    elif score["posture"] == "unknown" and two_minute:
        clock = "unknown"
    else:
        clock = "normal"

    return {
        "schema": ASSESSMENT_SCHEMA,
        "down": down, "distance": distance, "yardline": yardline,
        "quarter": quarter,
        "goal_to_go": goal_line or (red_zone and distance is not None
                                    and yardline is not None and yardline + distance >= 100),
        "red_zone": red_zone, "goal_line": goal_line,
        "backed_up": backed_up,
        "two_minute": two_minute,
        "phases": phases,
        "score": score,
        "clock_posture": clock,
        "conversion": {
            "needed_yards": distance,
            "short_yardage": short,
            "long_yardage": long_down,
        },
        "defense": _defensive_knowledge(sit, db=db, opponent_id=opponent_id),
        "objective": (
            "game-winning football: conversion and possession value in context, "
            "not generic completion probability"
        ),
    }


def _prior(name: str) -> float:
    return float(SITUATION_PRIORS[name]["delta"])


def play_situation_fit(play: str, assessment: Mapping[str, Any]) -> tuple[float, list[str]]:
    """How this play's family fits the assessed situation.

    Returns ``(delta, reasons)`` where every contribution names its prior in
    :data:`SITUATION_PRIORS`. Unknown down/distance contributes nothing.
    """
    phases = set(assessment.get("phases") or [])
    down = assessment.get("down")
    distance = assessment.get("distance")
    if down is None or distance is None:
        return 0.0, ["down/distance unknown"]
    run = is_run(play)
    family = _play_family(play)
    screen = family == "screen"
    deep = is_deep(play)
    score = assessment.get("score") or {}
    delta = 0.0
    reasons: list[str] = []

    def apply(name: str) -> None:
        nonlocal delta
        delta += _prior(name)
        reasons.append(f"{name}: {SITUATION_PRIORS[name]['rationale']}")

    if "third_or_fourth_long" in phases:
        if run:
            apply("long_down_run")
        elif screen:
            apply("long_down_screen")
        elif deep:
            apply("long_down_route_past_sticks")
    elif "short_yardage" in phases:
        if run:
            apply("short_yardage_run")
        if deep:
            apply("short_yardage_slow_shot")
    if "second_and_short" in phases and deep and not run:
        apply("second_short_upside")
    if "goal_line" in phases:
        if deep:
            apply("goal_line_compressed_deep")
        elif run or family in ("pass", "rpo") and not deep:
            apply("goal_line_run_or_quick")
    if "backed_up" in phases and (screen or deep):
        apply("backed_up_risk")
    if "two_minute" in phases and run and distance >= 4:
        if score.get("posture") == "trail":
            apply("two_minute_trailing_run")
        elif score.get("posture") in ("protect", "prevent"):
            apply("two_minute_leading_run")
        # Unknown score: no assumption about trailing or leading.
    if "protect_lead_late" in phases and "two_minute" not in phases:
        if run:
            apply("protect_lead_possession")
        elif deep:
            apply("protect_lead_low_percentage_shot")
    if "trailing_late" in phases and "two_minute" not in phases and run and distance >= 6:
        apply("trailing_late_clock_cost")
    return round(delta, 6), reasons or ["normal situation"]


# ---------------------------------------------------------------------------
# Pregame planning grid: the designer evaluates every formation across this
# labeled range of game states instead of three generic open-field checks.
# Weights reflect approximate share/leverage of offensive snaps, a clearly
# labeled planning prior rather than a learned distribution.
# ---------------------------------------------------------------------------
PREGAME_SITUATION_GRID: tuple[dict[str, Any], ...] = (
    {"name": "open_first_down", "down": 1, "distance": 10, "yardline": 35,
     "zone": "open", "weight": 1.0},
    {"name": "second_and_medium", "down": 2, "distance": 6, "yardline": 50,
     "zone": "open", "weight": 0.9},
    {"name": "second_and_long", "down": 2, "distance": 9, "yardline": 30,
     "zone": "open", "weight": 0.6},
    {"name": "third_and_medium", "down": 3, "distance": 5, "yardline": 45,
     "zone": "open", "weight": 0.9},
    {"name": "third_and_long", "down": 3, "distance": 10, "yardline": 30,
     "zone": "open", "weight": 0.8},
    {"name": "short_yardage", "down": 3, "distance": 1, "yardline": 55,
     "zone": "open", "weight": 0.7},
    {"name": "red_zone", "down": 1, "distance": 10, "yardline": 85,
     "zone": "rz", "weight": 0.7, "red_zone": True},
    {"name": "goal_line", "down": 2, "distance": 2, "yardline": 97,
     "zone": "gl", "weight": 0.5, "red_zone": True, "goal_line": True},
    {"name": "backed_up", "down": 1, "distance": 10, "yardline": 5,
     "zone": "open", "weight": 0.4},
    {"name": "two_minute_trailing", "down": 2, "distance": 8, "yardline": 40,
     "zone": "open", "weight": 0.6, "two_minute": True,
     "score_us": 17, "score_them": 21},
    {"name": "two_minute_protect", "down": 1, "distance": 10, "yardline": 60,
     "zone": "open", "weight": 0.4, "two_minute": True,
     "score_us": 24, "score_them": 17},
)


def grid_cell_assessment(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Build the same structured assessment for a synthetic pregame grid cell."""
    from cfb_coach.situation import Situation

    sit = Situation(raw=f"pregame:{cell['name']}")
    sit.down = cell.get("down")
    sit.distance = cell.get("distance")
    sit.yardline = cell.get("yardline")
    sit.red_zone = bool(cell.get("red_zone"))
    sit.goal_line = bool(cell.get("goal_line"))
    sit.two_minute = bool(cell.get("two_minute"))
    if cell.get("score_us") is not None:
        sit.score_us = int(cell["score_us"])
        sit.score_them = int(cell["score_them"])
    return assess_situation(sit)


def play_fits_grid_zone(play: str, cell: Mapping[str, Any]) -> bool:
    """Catalog zone eligibility for a pregame cell (open / rz / gl)."""
    return zone_fit(play, str(cell.get("zone") or "open"))


def concept_of(play: str) -> str:
    return _play_concept(play)
