"""Reusable football situation evaluator for Madden / CFB offense decisions.

Separates true observations from inferred tendencies and unknowns. Scores are
explicit, testable priors — never hidden bonuses or fabricated coverage certainty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from cfb_coach.madden.catalog import is_deep, is_run
from cfb_coach.madden.model.experimental_model import _play_concept, _play_family
from cfb_coach.madden.playcaller import coverage_class, situation_key

VERSION = "football_situation.v1"

# Explicit, source-backed strategic priors (not causal claims).
PRIOR_THIRD_LONG_RUN = -0.45
PRIOR_THIRD_LONG_SCREEN = -0.09
PRIOR_THIRD_LONG_DEEP = 0.025
PRIOR_SHORT_YARDAGE_RUN = 0.025
PRIOR_SHORT_YARDAGE_DEEP = -0.035
PRIOR_TWO_MINUTE_TRAIL_RUN = -0.07
PRIOR_TWO_MINUTE_LEAD_RUN = 0.04
PRIOR_LATE_LEAD_DEEP = -0.03
PRIOR_LATE_TRAIL_DEEP = 0.02
PRIOR_GOAL_LINE_SCREEN = -0.04
PRIOR_PRESSURE_QUICK = 0.03
PRIOR_PRESSURE_DEEP = -0.04


@dataclass(frozen=True)
class ObservationCredibility:
    """What is known vs inferred vs unknown at decision time."""

    down_distance: str  # observed | unknown
    field_position: str
    score: str
    clock: str
    coverage: str  # live | last_snap_soft | tendency | unknown
    coverage_class: str | None
    pressure: str  # live | tendency | unknown
    roster: str  # verified | unavailable
    notes: tuple[str, ...] = ()


@dataclass
class FootballSituation:
    """Structured football context for play and adjustment scoring."""

    version: str = VERSION
    situation_bucket: str = "early_down"
    down: int | None = None
    distance: int | None = None
    yardline: int | None = None
    yards_to_marker: int | None = None
    goal_to_go: bool = False
    red_zone: bool = False
    goal_line: bool = False
    short_yardage: bool = False
    long_yardage: bool = False
    third_and_long: bool = False
    two_minute: bool = False
    score_us: int | None = None
    score_them: int | None = None
    score_diff: int | None = None
    possession_objective: str = "neutral"  # protect_lead | trail | neutral
    quarter: int | None = None
    clock_seconds: int | None = None
    timeouts_us: int | None = None
    timeouts_them: int | None = None
    coverage_hint: str | None = None
    coverage_source: str = "none"
    coverage_class: str | None = None
    credible_look: bool = False
    pressure_credible: bool = False
    credibility: ObservationCredibility = field(
        default_factory=lambda: ObservationCredibility(
            "unknown", "unknown", "unknown", "unknown", "unknown", None, "unknown", "unavailable",
        )
    )
    rationale: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def evaluate_situation(
    sit: Any,
    *,
    repeated_live_look: bool = False,
    opponent_tendencies: Mapping[str, Any] | None = None,
    roster_verified: bool = False,
) -> FootballSituation:
    """Build an explicit situation record from a live / replay Situation object."""
    try:
        down = int(getattr(sit, "down", None)) if getattr(sit, "down", None) is not None else None
    except (TypeError, ValueError):
        down = None
    try:
        distance = int(getattr(sit, "distance", None)) if getattr(sit, "distance", None) is not None else None
    except (TypeError, ValueError):
        distance = None
    try:
        yardline = int(getattr(sit, "yardline", None)) if getattr(sit, "yardline", None) is not None else None
    except (TypeError, ValueError):
        yardline = None

    red_zone = bool(getattr(sit, "red_zone", False))
    goal_line = bool(getattr(sit, "goal_line", False))
    two_minute = bool(getattr(sit, "two_minute", False))
    score_us = getattr(sit, "score_us", None)
    score_them = getattr(sit, "score_them", None)
    try:
        score_us_i = int(score_us) if score_us is not None else None
        score_them_i = int(score_them) if score_them is not None else None
    except (TypeError, ValueError):
        score_us_i = score_them_i = None
    score_diff = (
        score_us_i - score_them_i
        if score_us_i is not None and score_them_i is not None else None
    )
    if score_diff is None:
        objective = "neutral"
    elif score_diff > 0:
        objective = "protect_lead"
    elif score_diff < 0:
        objective = "trail"
    else:
        objective = "neutral"

    short = bool(down in (3, 4) and distance is not None and 0 < distance <= 2)
    long_y = bool(down in (3, 4) and distance is not None and distance >= 7)
    third_long = bool(down in (3, 4) and distance is not None and distance >= 7 and not goal_line)
    goal_to_go = bool(goal_line or (distance is not None and yardline is not None and distance >= yardline > 0 and yardline <= 10))

    cov = getattr(sit, "coverage_hint", None)
    source = getattr(sit, "coverage_source", None) or "none"
    cls = coverage_class(cov) if cov else None
    credible = bool(cls) and (source == "live" or repeated_live_look)
    pressure = bool(credible and cls == "pressure")

    tendencies = dict(opponent_tendencies or {})
    tendency_pressure = float(tendencies.get("pressure_rate") or 0.0)
    tendency_n = int(tendencies.get("n_looks") or 0)
    # Tendency is never treated as current coverage certainty.
    if not pressure and tendency_n >= 4 and tendency_pressure >= 0.55:
        pressure_note = "tendency_pressure_elevated"
    else:
        pressure_note = None

    if source == "live" and cls:
        cov_cred = "live"
    elif source == "last" and cls:
        cov_cred = "last_snap_soft"
    elif tendency_n >= 4 and tendencies.get("modal_coverage"):
        cov_cred = "tendency"
    else:
        cov_cred = "unknown"

    try:
        bucket = situation_key(sit) if sit is not None else "early_down"
    except Exception:  # noqa: BLE001 — probes may omit Situation helpers
        if goal_line:
            bucket = "goal_line"
        elif red_zone:
            bucket = "red_zone"
        elif two_minute:
            bucket = "two_minute"
        elif long_y:
            bucket = "long_yardage"
        elif short:
            bucket = "short_yardage"
        else:
            bucket = "early_down"
    rationale = [
        f"bucket={bucket}",
        f"objective={objective}",
        f"coverage_credibility={cov_cred}",
    ]
    if pressure_note:
        rationale.append(pressure_note)
    if not credible and cls:
        rationale.append("prior_look_is_not_current_coverage")

    credibility = ObservationCredibility(
        down_distance="observed" if down is not None and distance is not None else "unknown",
        field_position="observed" if yardline is not None or red_zone or goal_line else "unknown",
        score="observed" if score_diff is not None else "unknown",
        clock="observed" if two_minute or getattr(sit, "clock_seconds", None) is not None else "unknown",
        coverage=cov_cred,
        coverage_class=cls if cov_cred == "live" else (cls if cov_cred == "last_snap_soft" else None),
        pressure="live" if pressure else ("tendency" if pressure_note else "unknown"),
        roster="verified" if roster_verified else "unavailable",
        notes=tuple(rationale),
    )

    return FootballSituation(
        situation_bucket=bucket,
        down=down,
        distance=distance,
        yardline=yardline,
        yards_to_marker=distance,
        goal_to_go=goal_to_go,
        red_zone=red_zone,
        goal_line=goal_line,
        short_yardage=short,
        long_yardage=long_y,
        third_and_long=third_long,
        two_minute=two_minute,
        score_us=score_us_i,
        score_them=score_them_i,
        score_diff=score_diff,
        possession_objective=objective,
        quarter=getattr(sit, "quarter", None),
        clock_seconds=getattr(sit, "clock_seconds", None),
        timeouts_us=getattr(sit, "timeouts_us", None),
        timeouts_them=getattr(sit, "timeouts_them", None),
        coverage_hint=str(cov) if cov else None,
        coverage_source=source,
        coverage_class=cls,
        credible_look=credible,
        pressure_credible=pressure,
        credibility=credibility,
        rationale=rationale,
    )


def play_situation_fit(play: str, situation: FootballSituation) -> tuple[float, list[str]]:
    """Explicit, labeled suitability prior for a play given the situation."""
    if not play:
        return 0.0, ["empty play"]
    run = is_run(play)
    family = _play_family(play)
    screen = family == "screen"
    deep = is_deep(play)
    delta = 0.0
    reasons: list[str] = []

    if situation.third_and_long:
        if run:
            delta += PRIOR_THIRD_LONG_RUN
            reasons.append("third/fourth-and-long: run conversion risk")
        if screen:
            delta += PRIOR_THIRD_LONG_SCREEN
            reasons.append("third/fourth-and-long: screen often short of sticks")
        if not run and not screen and deep:
            delta += PRIOR_THIRD_LONG_DEEP
            reasons.append("third/fourth-and-long: deep concept may reach marker")
    elif situation.short_yardage:
        if run:
            delta += PRIOR_SHORT_YARDAGE_RUN
            reasons.append("short-yardage: run conversion option")
        if deep:
            delta += PRIOR_SHORT_YARDAGE_DEEP
            reasons.append("short-yardage: long-developing route less suitable")

    if situation.two_minute:
        if situation.possession_objective == "trail" and run:
            delta += PRIOR_TWO_MINUTE_TRAIL_RUN
            reasons.append("two-minute trailing: clock-friendly pass preferred")
        elif situation.possession_objective == "protect_lead" and run:
            delta += PRIOR_TWO_MINUTE_LEAD_RUN
            reasons.append("two-minute lead: run can consume clock")

    if situation.possession_objective == "protect_lead" and situation.quarter in (4, None) and deep:
        # Soft clock-security prior only when score is known as a lead.
        if situation.score_diff is not None and situation.score_diff >= 8:
            delta += PRIOR_LATE_LEAD_DEEP
            reasons.append("late lead: deep shot increases possession risk")
    if situation.possession_objective == "trail" and situation.two_minute and deep:
        delta += PRIOR_LATE_TRAIL_DEEP
        reasons.append("trailing late: explosive concept has conversion upside")

    if situation.goal_line and screen:
        delta += PRIOR_GOAL_LINE_SCREEN
        reasons.append("goal line: compressed spacing hurts screens")

    if situation.pressure_credible:
        concept = _play_concept(play)
        if concept in ("slant", "quick", "screen", "mesh") or family == "screen":
            delta += PRIOR_PRESSURE_QUICK
            reasons.append("credible pressure: quick-game concept preferred")
        if deep and not run:
            delta += PRIOR_PRESSURE_DEEP
            reasons.append("credible pressure: deep timing routes are riskier")

    if not reasons:
        reasons.append("neutral situation fit")
    return round(delta, 5), reasons


def situation_coverage_matrix() -> tuple[str, ...]:
    """Football situations the pregame planner must cover."""
    return (
        "normal_early_down",
        "short_yardage",
        "third_and_long",
        "red_zone",
        "goal_line",
        "backed_up",
        "two_minute",
        "protect_lead",
        "trail",
        "pressure_response",
        "man_response",
        "zone_response",
    )


def synthetic_probe_situations() -> list[dict[str, Any]]:
    """Deterministic probes used by pregame formation portfolio scoring."""
    return [
        {"label": "normal_early_down", "down": 1, "distance": 10, "yardline": 35,
         "red_zone": False, "goal_line": False, "two_minute": False},
        {"label": "short_yardage", "down": 3, "distance": 1, "yardline": 40,
         "red_zone": False, "goal_line": False, "two_minute": False},
        {"label": "third_and_long", "down": 3, "distance": 10, "yardline": 35,
         "red_zone": False, "goal_line": False, "two_minute": False},
        {"label": "red_zone", "down": 2, "distance": 7, "yardline": 12,
         "red_zone": True, "goal_line": False, "two_minute": False},
        {"label": "goal_line", "down": 1, "distance": 3, "yardline": 3,
         "red_zone": True, "goal_line": True, "two_minute": False},
        {"label": "backed_up", "down": 1, "distance": 10, "yardline": 5,
         "red_zone": False, "goal_line": False, "two_minute": False},
        {"label": "two_minute", "down": 2, "distance": 8, "yardline": 45,
         "red_zone": False, "goal_line": False, "two_minute": True,
         "score_us": 14, "score_them": 21},
        {"label": "protect_lead", "down": 1, "distance": 10, "yardline": 40,
         "red_zone": False, "goal_line": False, "two_minute": True,
         "score_us": 24, "score_them": 17, "quarter": 4},
        {"label": "trail", "down": 3, "distance": 6, "yardline": 45,
         "red_zone": False, "goal_line": False, "two_minute": True,
         "score_us": 10, "score_them": 24, "quarter": 4},
        {"label": "pressure_response", "down": 2, "distance": 7, "yardline": 35,
         "red_zone": False, "goal_line": False, "two_minute": False,
         "coverage_hint": "blitz", "coverage_source": "live"},
        {"label": "man_response", "down": 1, "distance": 10, "yardline": 30,
         "red_zone": False, "goal_line": False, "two_minute": False,
         "coverage_hint": "man", "coverage_source": "live"},
        {"label": "zone_response", "down": 2, "distance": 8, "yardline": 40,
         "red_zone": False, "goal_line": False, "two_minute": False,
         "coverage_hint": "cover 3", "coverage_source": "live"},
    ]


def sit_from_probe(probe: Mapping[str, Any]) -> Any:
    """Minimal Situation-like namespace for portfolio probes."""
    from types import SimpleNamespace

    try:
        down = int(probe["down"]) if probe.get("down") is not None else None
    except (TypeError, ValueError):
        down = None
    try:
        distance = int(probe["distance"]) if probe.get("distance") is not None else None
    except (TypeError, ValueError):
        distance = None
    long_yardage = bool(down in (3, 4) and distance is not None and distance >= 7)
    short_yardage = bool(distance is not None and distance <= 2)
    return SimpleNamespace(
        down=down,
        distance=distance,
        yardline=probe.get("yardline"),
        red_zone=bool(probe.get("red_zone")),
        goal_line=bool(probe.get("goal_line")),
        two_minute=bool(probe.get("two_minute")),
        long_yardage=long_yardage,
        short_yardage=short_yardage,
        score_us=probe.get("score_us"),
        score_them=probe.get("score_them"),
        quarter=probe.get("quarter"),
        coverage_hint=probe.get("coverage_hint"),
        coverage_source=probe.get("coverage_source") or "none",
        clock_seconds=probe.get("clock_seconds"),
        timeouts_us=probe.get("timeouts_us"),
        timeouts_them=probe.get("timeouts_them"),
        extras={},
    )
